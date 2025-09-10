from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io
import os
import json
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import uuid
import requests

load_dotenv()

# === Supabase Initialization ===
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# Debug environment variables
print(f"🔍 SUPABASE_URL: {supabase_url}")
print(f"🔍 SUPABASE_KEY: {'*' * 20 if supabase_key else 'None'}")

if not supabase_url:
    raise ValueError("SUPABASE_URL environment variable is not set!")
if not supabase_key:
    raise ValueError("SUPABASE_KEY environment variable is not set!")

supabase: Client = create_client(supabase_url, supabase_key)

# === Debug info to check Railway runtime ===
print("🚀 TensorFlow version:", tf.__version__)
print("🖥️  Available devices:", tf.config.list_physical_devices())

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

app = Flask(__name__)

# === CORS Configuration ===
CORS(app, resources={
    r"/*": {
        "origins": ["*"],  # Allow all origins in development
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Model variables (loaded on demand)
fire_model = None
structure_model = None
smoke_model = None

FIRE_CLASSES = ['Fire', 'No Fire']
STRUCTURE_CLASSES = ['Concrete Building', 'Metal Structure', 'Wooden Houses']
SMOKE_CLASSES = ['high', 'low', 'medium']

def determine_alarm_level(houses):
    if houses is None:
        return "Unknown - structure count not provided"
    
    if houses >= 640:
        return "General Alarm - Major area affected (~80 fire trucks)"
    elif houses >= 288:
        return "Task Force Delta - Significant part (~36 fire trucks)"
    elif houses >= 256:
        return "Task Force Charlie - Significant part (~32 fire trucks)"
    elif houses >= 225:
        return "Task Force Bravo - ~28 fire trucks"
    elif houses >= 144:
        return "Task Force Alpha - ~24 fire trucks"
    elif houses >= 11:
        return "Fifth Alarm - 20 fire trucks"
    elif houses >= 9:
        return "Fourth Alarm - 16 fire trucks"
    elif houses >= 7:
        return "Third Alarm - 12 fire trucks"
    elif houses >= 5:
        return "Second Alarm - 8 fire trucks"
    elif houses >= 2:
        return "First Alarm - 4 fire trucks"
    elif houses >= 1:
        return "Under Control - Low fire risk"
    else:
        return "Fire Out - Fire has been neutralized"


def reverse_geocode(latitude, longitude):
    """
    Convert coordinates to human-readable address using Nominatim API
    Returns formatted address string or coordinates as fallback
    """
    try:
        # Using Nominatim (OpenStreetMap) - free geocoding service
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'format': 'json',
            'lat': latitude,
            'lon': longitude,
            'zoom': 18,
            'addressdetails': 1
        }
        headers = {
            'User-Agent': 'FireDetectionApp/1.0'  # Required by Nominatim
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            # Build address string in order of preference
            address_parts = []
            
            # Street number and street
            if address.get('house_number') and address.get('road'):
                address_parts.append(f"{address['house_number']} {address['road']}")
            elif address.get('road'):
                address_parts.append(address['road'])
            
            # Suburb, village, or neighbourhood
            if address.get('suburb'):
                address_parts.append(address['suburb'])
            elif address.get('village'):
                address_parts.append(address['village'])
            elif address.get('neighbourhood'):
                address_parts.append(address['neighbourhood'])
            
            # City, town, or municipality
            if address.get('city'):
                address_parts.append(address['city'])
            elif address.get('town'):
                address_parts.append(address['town'])
            elif address.get('municipality'):
                address_parts.append(address['municipality'])
            
            # State/Province
            if address.get('state'):
                address_parts.append(address['state'])
            elif address.get('province'):
                address_parts.append(address['province'])
            
            # Country
            if address.get('country'):
                address_parts.append(address['country'])
            
            # Join parts with commas
            if address_parts:
                return ', '.join(address_parts)
        
        # Fallback to coordinates if no address found
        return f"{latitude:.6f}, {longitude:.6f}"
        
    except Exception as e:
        print(f"Error in reverse geocoding: {str(e)}")
        # Return coordinates as fallback
        return f"{latitude:.6f}, {longitude:.6f}"

def parse_coordinates(location_string):
    """
    Parse coordinate string to extract latitude and longitude
    Returns tuple (latitude, longitude) or (None, None) if parsing fails
    """
    try:
        if not location_string:
            return None, None
        
        # Split by comma and clean up
        coords = [coord.strip() for coord in location_string.split(',')]
        
        if len(coords) >= 2:
            lat = float(coords[0])
            lng = float(coords[1])
            
            # Basic validation for valid coordinates
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                return lat, lng
        
        return None, None
    except (ValueError, TypeError):
        return None, None

def upload_image_to_supabase(file, folder="fire_reports"):
    """
    Upload image to Supabase Storage bucket
    Returns the public URL of the uploaded image
    """
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = f"{folder}/{unique_filename}"
        
        # Reset file pointer to beginning
        file.stream.seek(0)
        file_data = file.read()
        
        # Upload to Supabase Storage
        response = supabase.storage.from_("fire_reports").upload(file_path, file_data)
        
        # Check if upload was successful
        print(f"🔍 Upload response: {response}")
        
        # If there's an error in the response, it will have an 'error' key
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Upload failed: {response.error}")
        
        # Get public URL
        public_url = supabase.storage.from_("fire_reports").get_public_url(file_path)
        
        return public_url
    except Exception as e:
        print(f"Error uploading to Supabase Storage: {str(e)}")
        raise e

def save_report_to_supabase(image_url, prediction_json, geotag_location=None, cause_of_fire=None, user_id=None, user_name=None):
    """
    Save report with both coordinates and resolved address, including status
    """
    manila_tz = timezone(timedelta(hours=8))
    local_time = datetime.now(manila_tz)
    formatted_time = local_time.strftime("%B %d %I:%M %p").replace(" 0", " ").replace("AM", "am").replace("PM", "pm")
    
    # Parse coordinates from geotag_location
    latitude, longitude = parse_coordinates(geotag_location)
    
    # Get human-readable address if coordinates are valid
    resolved_address = None
    if latitude is not None and longitude is not None:
        print(f"🗺️ Resolving address for coordinates: {latitude}, {longitude}")
        resolved_address = reverse_geocode(latitude, longitude)
        print(f"🏠 Resolved address: {resolved_address}")
    
    # Determine status based on fire prediction
    # Default status is "On Going" for any fire detection
    fire_prediction = prediction_json.get('prediction', '').lower()
    if fire_prediction == 'fire':
        status = 'On Going'
    elif fire_prediction == 'no fire':
        # Check if alarm level indicates fire is out
        alarm_level = prediction_json.get('recommended_alarm_level', '').lower()
        if 'fireout' in alarm_level or 'neutralized' in alarm_level:
            status = 'Fire Out'
        else:
            status = 'On Going'  # Default to On Going even for "No Fire" initially
    else:
        status = 'On Going'  # Default fallback
    
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formatted_timestamp": formatted_time,
        "image_url": image_url,
        # Store raw coordinates for geo-queries
        "latitude": latitude,
        "longitude": longitude,
        "geotag_location": geotag_location,  # Keep original coordinate string
        # Store resolved human-readable address
        "address": resolved_address,
        "cause_of_fire": cause_of_fire[:200] if cause_of_fire else None,
        "reporter": user_name,
        "reporterId": user_id,
        # Add status field
        "status": status,
        **prediction_json
    }
    
    result = supabase.table("fire_reports").insert(data).execute()
    print(f"💾 Saved report to database with address: {resolved_address} and status: {status}")
    return result

@app.route('/')
def home():
    return "Fire Detection API is running! (Fire + Structure + Smoke detection active)"

@app.route('/predict', methods=['POST'])
def predict():
    global fire_model, structure_model, smoke_model
    
    # Lazy load models
    if fire_model is None:
        print("Loading fire detection model...")
        fire_model = load_model("fire_mobilenet_model.keras")
    if structure_model is None:
        print("Loading structure classification model...")
        structure_model = load_model("structure_material_classifier.keras")
    if smoke_model is None:
        print("Loading smoke detection model...")
        smoke_model = load_model("smoke_balanced_mobilenetv2_model_fixed_final.keras")

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    num_structures = request.form.get('number_of_structures_on_fire')
    geotag_location = request.form.get('geotag_location')  # Raw coordinates
    cause_of_fire = request.form.get('cause_of_fire')
    user_id = request.form.get('user_id')
    user_name = request.form.get('user_name')
    
    try:
        num_structures = int(num_structures)
    except (TypeError, ValueError):
        num_structures = None

    try:
        # ==== 1️⃣ Upload image to Supabase Storage ====
        image_url = upload_image_to_supabase(file)

        # ==== 2️⃣ Prepare image for prediction ====
        file.stream.seek(0)  # reset pointer since upload read it
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Make predictions
        fire_pred = fire_model.predict(image, verbose=0)[0]
        fire_result = FIRE_CLASSES[np.argmax(fire_pred)]
        fire_confidence = float(np.max(fire_pred)) * 100

        structure_pred = structure_model.predict(image, verbose=0)[0]
        structure_result = STRUCTURE_CLASSES[np.argmax(structure_pred)]

        smoke_pred = smoke_model.predict(image, verbose=0)[0]
        smoke_result = SMOKE_CLASSES[np.argmax(smoke_pred)]
        smoke_confidence = float(np.max(smoke_pred)) * 100

        alarm_level = determine_alarm_level(num_structures)

        # ==== 3️⃣ Save to Supabase with coordinates AND address ====
        prediction_data = {
            'prediction': fire_result,
            'confidence': f"{fire_confidence:.2f}%",
            'structure': structure_result,
            'number_of_structures_on_fire': num_structures,
            'recommended_alarm_level': alarm_level,
            'smoke_intensity': smoke_result,
            'smoke_confidence': f"{smoke_confidence:.2f}%"
        }
        
        # Save report (this will now include both coordinates and resolved address)
        save_report_to_supabase(image_url, prediction_data, geotag_location, cause_of_fire, user_id, user_name)

        # ==== 4️⃣ Parse coordinates for response ====
        latitude, longitude = parse_coordinates(geotag_location)
        resolved_address = None
        if latitude is not None and longitude is not None:
            resolved_address = reverse_geocode(latitude, longitude)

        # ==== 5️⃣ Return enhanced response ====   
        return jsonify({
            'image_url': image_url,
            'geotag_location': geotag_location,  # Raw coordinates
            'latitude': latitude,                 # Parsed latitude
            'longitude': longitude,               # Parsed longitude
            'address': resolved_address,          # Human-readable address
            'cause_of_fire': cause_of_fire,
            **prediction_data
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_reports', methods=['GET'])
def get_reports():
    try:
        # Select all fields including the new latitude, longitude, and address fields
        response = supabase.table("fire_reports").select("*").execute()
        reports_list = response.data
        
        # Sort by timestamp if needed (Supabase can also sort in SQL)
        reports_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Process each report to ensure address is available
        for report in reports_list:
            # If address is missing but coordinates exist, resolve it
            if not report.get('address') and report.get('latitude') and report.get('longitude'):
                try:
                    resolved_address = reverse_geocode(report['latitude'], report['longitude'])
                    report['address'] = resolved_address
                    
                    # Update the database with the resolved address
                    supabase.table("fire_reports").update({
                        'address': resolved_address
                    }).eq('id', report['id']).execute()
                    
                    print(f"🔄 Updated missing address for report {report['id']}: {resolved_address}")
                except Exception as e:
                    print(f"Error resolving missing address for report {report['id']}: {str(e)}")
                    report['address'] = f"{report['latitude']:.6f}, {report['longitude']:.6f}"
            
            # Ensure backwards compatibility - if old format exists without lat/lng
            elif not report.get('latitude') and report.get('geotag_location'):
                lat, lng = parse_coordinates(report['geotag_location'])
                if lat is not None and lng is not None:
                    # Resolve address and update database
                    try:
                        resolved_address = reverse_geocode(lat, lng)
                        
                        # Update database with parsed coordinates and resolved address
                        supabase.table("fire_reports").update({
                            'latitude': lat,
                            'longitude': lng,
                            'address': resolved_address
                        }).eq('id', report['id']).execute()
                        
                        # Update current response
                        report['latitude'] = lat
                        report['longitude'] = lng
                        report['address'] = resolved_address
                        
                        print(f"🔄 Migrated legacy report {report['id']} with coordinates and address")
                    except Exception as e:
                        print(f"Error migrating legacy report {report['id']}: {str(e)}")
        
        return jsonify(reports_list)
    except Exception as e:
        print(f"Error retrieving reports: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_report/<report_id>', methods=['PUT'])
def update_report(report_id):
    try:
        # Check if this is a multipart request (image update)
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle image update with AI re-processing
            if 'image' not in request.files:
                return jsonify({'error': 'No image provided for update'}), 400
            
            file = request.files['image']
            cause_of_fire = request.form.get('cause_of_fire')
            num_structures = request.form.get('number_of_structures_on_fire')
            status = request.form.get('status')  # NEW: Get status from form data
            
            # Handle location updates in multipart requests
            geotag_location = request.form.get('geotag_location')
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')
            address = request.form.get('address')
            
            try:
                num_structures = int(num_structures) if num_structures else None
            except (TypeError, ValueError):
                num_structures = None
            
            # Parse latitude and longitude if provided
            try:
                latitude = float(latitude) if latitude else None
                longitude = float(longitude) if longitude else None
            except (TypeError, ValueError):
                latitude = None
                longitude = None
            
            # Upload new image
            image_url = upload_image_to_supabase(file)
            
            # Re-run AI models on new image
            file.stream.seek(0)
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            image = image.resize((224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)
            
            # Load models if needed
            global fire_model, structure_model, smoke_model
            if fire_model is None:
                fire_model = load_model("fire_mobilenet_model.keras")
            if structure_model is None:
                structure_model = load_model("structure_material_classifier.keras")
            if smoke_model is None:
                smoke_model = load_model("smoke_balanced_mobilenetv2_model_fixed_final.keras")
            
            # Make new predictions
            fire_pred = fire_model.predict(image, verbose=0)[0]
            fire_result = FIRE_CLASSES[np.argmax(fire_pred)]
            fire_confidence = float(np.max(fire_pred)) * 100
            
            structure_pred = structure_model.predict(image, verbose=0)[0]
            structure_result = STRUCTURE_CLASSES[np.argmax(structure_pred)]
            
            smoke_pred = smoke_model.predict(image, verbose=0)[0]
            smoke_result = SMOKE_CLASSES[np.argmax(smoke_pred)]
            smoke_confidence = float(np.max(smoke_pred)) * 100
            
            alarm_level = determine_alarm_level(num_structures)
            
            # Determine status - use provided status if valid, otherwise auto-determine
            valid_statuses = ['On Going', 'Under Control', 'Fire Out', 'False Alarm', 'Cancelled']
            if status and status in valid_statuses:
                final_status = status
            else:
                # Auto-determine based on new prediction
                if fire_result == 'Fire':
                    final_status = 'On Going'
                elif fire_result == 'No Fire':
                    if 'fireout' in alarm_level.lower() or 'neutralized' in alarm_level.lower():
                        final_status = 'Fire Out'
                    else:
                        final_status = 'On Going'  # Default to On Going
                else:
                    final_status = 'On Going'
            
            # Update data with new predictions and location
            update_data = {
                'image_url': image_url,
                'cause_of_fire': cause_of_fire[:200] if cause_of_fire else None,
                'number_of_structures_on_fire': num_structures,
                'prediction': fire_result,
                'confidence': f"{fire_confidence:.2f}%",
                'structure': structure_result,
                'recommended_alarm_level': alarm_level,
                'smoke_intensity': smoke_result,
                'smoke_confidence': f"{smoke_confidence:.2f}%",
                'status': final_status  # NEW: Include status
            }
            
            # Add location fields if provided
            if geotag_location:
                update_data['geotag_location'] = geotag_location
            if latitude is not None:
                update_data['latitude'] = latitude
            if longitude is not None:
                update_data['longitude'] = longitude
            if address:
                update_data['address'] = address
                
        else:
            # Handle text-only updates (JSON)
            data = request.get_json()
            update_data = {}
            
            if 'cause_of_fire' in data:
                update_data['cause_of_fire'] = data['cause_of_fire'][:200]
            
            if 'number_of_structures_on_fire' in data:
                try:
                    num_structures = int(data['number_of_structures_on_fire'])
                    update_data['number_of_structures_on_fire'] = num_structures
                    update_data['recommended_alarm_level'] = determine_alarm_level(num_structures)
                except (TypeError, ValueError):
                    update_data['number_of_structures_on_fire'] = None
                    update_data['recommended_alarm_level'] = "Unknown - structure count not provided"
            
            # NEW: Handle status updates in JSON requests
            if 'status' in data:
                valid_statuses = ['On Going', 'Under Control', 'Fire Out', 'False Alarm']
                if data['status'] in valid_statuses:
                    update_data['status'] = data['status']
                else:
                    return jsonify({'error': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'}), 400
            
            # Handle location updates in JSON requests
            if 'geotag_location' in data:
                update_data['geotag_location'] = data['geotag_location']
            
            if 'latitude' in data:
                try:
                    update_data['latitude'] = float(data['latitude'])
                except (TypeError, ValueError):
                    update_data['latitude'] = None
            
            if 'longitude' in data:
                try:
                    update_data['longitude'] = float(data['longitude'])
                except (TypeError, ValueError):
                    update_data['longitude'] = None
            
            if 'address' in data:
                update_data['address'] = data['address']
        
        # Execute update
        result = supabase.table("fire_reports").update(update_data).eq('id', report_id).execute()
        
        if result.data:
            print(f"✅ Updated report {report_id} with status: {update_data.get('status', 'unchanged')}")
        
        # Return updated record
        updated = supabase.table("fire_reports").select("*").eq('id', report_id).execute()
        return jsonify(updated.data[0] if updated.data else {})
    except Exception as e:
        print(f"Error updating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_report/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    try:
        supabase.table("fire_reports").delete().eq('id', report_id).execute()
        return jsonify({'message': 'Report deleted successfully'})
    except Exception as e:
        print(f"Error deleting report: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
# ENHANCED BACKEND CANCEL REPORT ENDPOINT
@app.route('/cancel_report/<report_id>', methods=['POST'])
def cancel_report(report_id):
    """
    Dedicated endpoint for cancelling a fire report
    Sets the report status to 'Cancelled' and adds a cancellation timestamp
    """
    try:
        print(f"🚫 Attempting to cancel report: {report_id}")
        
        # Validate report_id
        if not report_id or report_id.strip() == '':
            return jsonify({'error': 'Invalid report ID provided'}), 400
        
        # Check if report exists first
        existing_report = supabase.table("fire_reports").select("*").eq('id', report_id).execute()
        
        if not existing_report.data:
            print(f"❌ Report {report_id} not found")
            return jsonify({'error': 'Report not found'}), 404
        
        report = existing_report.data[0]
        current_status = report.get('status', '')
        
        # Check if already cancelled
        if current_status == 'Cancelled':
            print(f"⚠️ Report {report_id} is already cancelled")
            return jsonify({
                'message': 'Report is already cancelled',
                'cancelled_at': report.get('cancellation_timestamp', 'Previously cancelled'),
                'report': report
            }), 200
        
        # Check if report is in a final state that shouldn't be cancelled
        if current_status == 'Fire Out':
            return jsonify({'error': 'Cannot cancel a report that has been marked as Fire Out'}), 400
        
        # Get current timestamp for cancellation
        manila_tz = timezone(timedelta(hours=8))
        local_time = datetime.now(manila_tz)
        cancellation_time = local_time.strftime("%B %d %I:%M %p").replace(" 0", " ").replace("AM", "am").replace("PM", "pm")
        
        # Update status to Cancelled and add cancellation details
        update_data = {
            'status': 'Cancelled',
            'cancelled_at': datetime.now(timezone.utc).isoformat(),
            'cancellation_timestamp': cancellation_time
        }
        
        print(f"📝 Updating report {report_id} with cancellation data: {update_data}")
        
        # Execute update
        result = supabase.table("fire_reports").update(update_data).eq('id', report_id).execute()
        
        if not result.data:
            print(f"❌ Failed to update report {report_id} - no data returned")
            return jsonify({'error': 'Failed to cancel report - update returned no data'}), 500
        
        print(f"✅ Successfully cancelled report {report_id} at {cancellation_time}")
        
        # Return updated record
        updated = supabase.table("fire_reports").select("*").eq('id', report_id).execute()
        updated_report = updated.data[0] if updated.data else {}
        
        return jsonify({
            'message': 'Report cancelled successfully',
            'cancelled_at': cancellation_time,
            'report': updated_report,
            'status': 'success'
        }), 200
        
    except ValueError as e:
        print(f"❌ ValueError cancelling report {report_id}: {str(e)}")
        return jsonify({'error': f'Invalid report ID format: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Exception cancelling report {report_id}: {str(e)}")
        return jsonify({
            'error': f'Failed to cancel report: {str(e)}',
            'report_id': report_id
        }), 500

@app.route('/reports_near', methods=['GET'])
def get_reports_near():
    """
    New endpoint: Get reports within a certain radius of given coordinates
    Query params: lat, lng, radius_km (default 10km)
    """
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        radius_km = float(request.args.get('radius_km', 10))
        
        # Simple bounding box calculation (approximate)
        # 1 degree ≈ 111 km
        lat_delta = radius_km / 111.0
        lng_delta = radius_km / (111.0 * abs(np.cos(np.radians(lat))))
        
        min_lat = lat - lat_delta
        max_lat = lat + lat_delta
        min_lng = lng - lng_delta
        max_lng = lng + lng_delta
        
        # Query reports within bounding box
        response = supabase.table("fire_reports").select("*").gte('latitude', min_lat).lte('latitude', max_lat).gte('longitude', min_lng).lte('longitude', max_lng).execute()
        
        # Filter by actual distance for more accuracy
        nearby_reports = []
        for report in response.data:
            if report.get('latitude') and report.get('longitude'):
                # Calculate distance using Haversine formula
                report_lat = report['latitude']
                report_lng = report['longitude']
                
                # Haversine distance calculation
                dlat = np.radians(report_lat - lat)
                dlng = np.radians(report_lng - lng)
                a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(report_lat)) * np.sin(dlng/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                distance_km = 6371 * c  # Earth radius in km
                
                if distance_km <= radius_km:
                    report['distance_km'] = round(distance_km, 2)
                    nearby_reports.append(report)
        
        # Sort by distance
        nearby_reports.sort(key=lambda x: x.get('distance_km', 0))
        
        return jsonify(nearby_reports)
    except (TypeError, ValueError) as e:
        return jsonify({'error': 'Invalid coordinates provided'}), 400
    except Exception as e:
        print(f"Error getting nearby reports: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_report_status/<report_id>', methods=['PATCH'])
def update_report_status(report_id):
    """
    Dedicated endpoint for updating just the status of a fire report
    Useful for admin dashboard quick status changes
    """
    try:
        data = request.get_json()
        
        if 'status' not in data:
            return jsonify({'error': 'Status field is required'}), 400
        
        new_status = data['status']
        valid_statuses = ['On Going', 'Under Control', 'Fire Out', 'False Alarm']
        
        if new_status not in valid_statuses:
            return jsonify({'error': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'}), 400
        
        # Update only the status
        update_data = {'status': new_status}
        
        # Execute update
        result = supabase.table("fire_reports").update(update_data).eq('id', report_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Report not found'}), 404
        
        print(f"✅ Updated report {report_id} status to: {new_status}")
        
        # Return updated record
        updated = supabase.table("fire_reports").select("*").eq('id', report_id).execute()
        return jsonify({
            'message': f'Status updated successfully to {new_status}',
            'report': updated.data[0] if updated.data else {}
        })
        
    except Exception as e:
        print(f"Error updating report status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_status_options', methods=['GET'])
def get_status_options():
    """
    Return available status options for frontend dropdowns
    """
    return jsonify({
        'status_options': [
            {'value': 'On Going', 'label': 'On Going', 'color': '#ef4444'},
            {'value': 'Under Control', 'label': 'Under Control', 'color': '#f59e0b'},
            {'value': 'Fire Out', 'label': 'Fire Out', 'color': '#10b981'},
            {'value': 'False Alarm', 'label': 'False Alarm', 'color': '#6b7280'}
        ]
    })

@app.route('/update_final_alarm_level', methods=['POST'])
def update_final_alarm_level():
    """
    Dedicated endpoint for updating the final_fire_alarm_level of a fire report
    Used by admin dashboard for setting final alarm levels
    """
    try:
        data = request.get_json()
        
        if 'report_id' not in data:
            return jsonify({'error': 'report_id field is required'}), 400
        
        if 'final_alarm_level' not in data:
            return jsonify({'error': 'final_alarm_level field is required'}), 400
        
        report_id = data['report_id']
        final_alarm_level = data['final_alarm_level']
        
        # Valid alarm levels
        valid_alarm_levels = [
            '1st Alarm', '2nd Alarm', '3rd Alarm', '4th Alarm', '5th Alarm',
            'TASK FORCE ALPHA', 'TASK FORCE BRAVO', 'TASK FORCE CHARLIE', 
            'TASK FORCE DELTA', 'GENERAL ALARM'
        ]
        
        if final_alarm_level not in valid_alarm_levels:
            return jsonify({'error': f'Invalid alarm level. Must be one of: {", ".join(valid_alarm_levels)}'}), 400
        
        # Update only the final_fire_alarm_level
        update_data = {'final_fire_alarm_level': final_alarm_level}
        
        # Execute update
        result = supabase.table("fire_reports").update(update_data).eq('id', report_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Report not found'}), 404
        
        print(f"✅ Updated report {report_id} final alarm level to: {final_alarm_level}")
        
        # Return updated record
        updated = supabase.table("fire_reports").select("*").eq('id', report_id).execute()
        return jsonify({
            'message': f'Final alarm level updated successfully to {final_alarm_level}',
            'report': updated.data[0] if updated.data else {}
        })
        
    except Exception as e:
        print(f"Error updating final alarm level: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/update_report_status', methods=['POST'])
def update_report_status_post():
    """
    POST version of status update for consistency with final alarm level endpoint
    """
    try:
        data = request.get_json()
        
        if 'report_id' not in data:
            return jsonify({'error': 'report_id field is required'}), 400
        
        if 'status' not in data:
            return jsonify({'error': 'status field is required'}), 400
        
        report_id = data['report_id']
        new_status = data['status']
        valid_statuses = ['On Going', 'Fire Out']
        
        if new_status not in valid_statuses:
            return jsonify({'error': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'}), 400
        
        # Update only the status
        update_data = {'status': new_status}
        
        # Execute update
        result = supabase.table("fire_reports").update(update_data).eq('id', report_id).execute()
        
        if not result.data:
            return jsonify({'error': 'Report not found'}), 404
        
        print(f"✅ Updated report {report_id} status to: {new_status}")
        
        # Return updated record
        updated = supabase.table("fire_reports").select("*").eq('id', report_id).execute()
        return jsonify({
            'message': f'Status updated successfully to {new_status}',
            'report': updated.data[0] if updated.data else {}
        })
        
    except Exception as e:
        print(f"Error updating report status: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
        

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)