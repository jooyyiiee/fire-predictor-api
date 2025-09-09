import cloudinary
import cloudinary.uploader
import cloudinary.api
import os
#from dotenv import load_dotenv


# Don't use load_dotenv() in production - Railway provides env vars directly
#load_dotenv()  # Comment this out or remove it

# Use environment variables for security
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),  # Must be set as env var
    secure=True
)