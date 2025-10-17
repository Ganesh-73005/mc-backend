from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header, WebSocket, WebSocketDisconnect, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import json
from datetime import datetime, timedelta
from bson import ObjectId
import jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet
import base64
import qrcode
from io import BytesIO
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
uri = os.environ.get('MONGO_URL')
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client[os.environ.get('DB_NAME', 'foodies_circle')]

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"

# Fernet encryption setup
FERNET_KEY = os.environ.get('FERNET_KEY', Fernet.generate_key().decode())
fernet = Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Realtime connection manager for feed notifications
class FeedConnectionManager:
    def __init__(self):
        self.active_foodie_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_foodie_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        self.active_foodie_connections.pop(user_id, None)

    async def broadcast_new_post(self, post_id: str, post_type: str):
        message = json.dumps({
            "type": "new_post",
            "post_id": post_id,
            "post_type": post_type
        })
        stale_user_ids: List[str] = []
        for uid, ws in self.active_foodie_connections.items():
            try:
                await ws.send_text(message)
            except Exception:
                stale_user_ids.append(uid)
        for uid in stale_user_ids:
            self.disconnect(uid)

feed_manager = FeedConnectionManager()

# Helper functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_token(user_id: str, user_type: str) -> str:
    payload = {
        "user_id": user_id,
        "user_type": user_type,
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.replace("Bearer ", "")
    payload = verify_token(token)
    return payload

def encrypt_promo_code(promo_text: str, promoter_id: str, restaurant_id: str, post_id: str, dish_id: str = "") -> str:
    data = f"{promo_text}|{promoter_id}|{restaurant_id}|{post_id}|{dish_id}"
    encrypted = fernet.encrypt(data.encode())
    return base64.urlsafe_b64encode(encrypted).decode()

def decrypt_promo_code(encrypted_code: str) -> Dict:
    try:
        decoded = base64.urlsafe_b64decode(encrypted_code.encode())
        decrypted = fernet.decrypt(decoded).decode()
        parts = decrypted.split("|")
        return {
            "promo_text": parts[0],
            "promoter_id": parts[1],
            "restaurant_id": parts[2],
            "post_id": parts[3],
            "dish_id": parts[4] if len(parts) > 4 else ""
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid promo code")

def generate_qr_code_base64(data: str) -> str:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Pydantic Models
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    profile_name: str
    handle: str
    user_type: str
    avatar_base64: Optional[str] = None
    bio: Optional[str] = None
    restaurant_details: Optional[Dict] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    
class PromoDecryptRequest(BaseModel):
    promo_code_encrypted: str

class UserUpdate(BaseModel):
    profile_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_base64: Optional[str] = None
    restaurant_details: Optional[Dict] = None

class LocationUpdate(BaseModel):
    latitude: float
    longitude: float
    address: Optional[str] = None
    place_name: Optional[str] = None

class PostCreate(BaseModel):
    image_base64: str
    caption: str
    stars: Optional[int] = None
    restaurant_tagged_id: Optional[str] = None
    location: Optional[Dict] = None
    is_promotion_request: bool = False
    promotion_offer_idea: Optional[str] = None

class CommentCreate(BaseModel):
    text: str

class PromoApprove(BaseModel):
    promo_code_plain_text: str
    offer_description: str
    expiry_date: Optional[str] = None

class PromoRedeem(BaseModel):
    promo_code_encrypted: str
    redeemer_user_id: str

# Auth Endpoints
@api_router.post("/register")
async def register(user: UserRegister):
    # Check if user exists
    existing_user = db.users.find_one({"$or": [{"email": user.email}, {"handle": user.handle}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or handle already exists")
    
    # Create user
    user_dict = user.dict()
    user_dict["password_hash"] = hash_password(user.password)
    del user_dict["password"]
    user_dict["followers"] = []
    user_dict["following"] = []
    user_dict["created_at"] = datetime.utcnow()
    
    result = db.users.insert_one(user_dict)
    user_id = str(result.inserted_id)
    
    token = create_token(user_id, user.user_type)
    
    return {
        "token": token,
        "user_id": user_id,
        "user_type": user.user_type,
        "profile_name": user.profile_name,
        "handle": user.handle
    }

@api_router.post("/login")
async def login(credentials: UserLogin):
    user = db.users.find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_id = str(user["_id"])
    token = create_token(user_id, user["user_type"])
    
    return {
        "token": token,
        "user_id": user_id,
        "user_type": user["user_type"],
        "profile_name": user["profile_name"],
        "handle": user["handle"],
        "avatar_base64": user.get("avatar_base64")
    }

@api_router.get("/me")
async def get_me(current_user: Dict = Depends(get_current_user)):
    user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user["_id"] = str(user["_id"])
    user.pop("password_hash", None)
    return user

# User Endpoints
@api_router.get("/users/search")
async def search_users(q: str, filter_type: Optional[str] = None):
    query = {"$or": [
        {"profile_name": {"$regex": q, "$options": "i"}},
        {"handle": {"$regex": q, "$options": "i"}}
    ]}
    
    if filter_type:
        query["user_type"] = filter_type
    
    users = db.users.find(query).limit(20).to_list(20)
    
    for user in users:
        user["_id"] = str(user["_id"])
        user.pop("password_hash", None)
    
    return users

@api_router.get("/users/{user_id}")
async def get_user(user_id: str):
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user["_id"] = str(user["_id"])
    user.pop("password_hash", None)
    
    # Get post count
    post_count = db.posts.count_documents({"user_id": user_id})
    user["post_count"] = post_count
    
    return user

@api_router.post("/promos/decrypt")
async def decrypt_promo_details(request: PromoDecryptRequest, current_user: Dict = Depends(get_current_user)):
    try:
        decrypted = decrypt_promo_code(request.promo_code_encrypted)

        if decrypted.get("restaurant_id") != current_user.get("user_id"):
            raise HTTPException(status_code=403, detail="This promo is not valid for your restaurant.")

        promoter = db.users.find_one({"_id": ObjectId(decrypted["promoter_id"])}, {"profile_name": 1, "handle": 1})
        restaurant = db.users.find_one({"_id": ObjectId(decrypted["restaurant_id"])}, {"profile_name": 1})
        post = db.posts.find_one({"_id": ObjectId(decrypted["post_id"])}, {"caption": 1})
        promo_info = db.promocodes.find_one({"code_encrypted": request.promo_code_encrypted}, {"offer_description": 1, "expiry_date": 1})

        if not promoter or not restaurant or not post or not promo_info:
            raise HTTPException(status_code=404, detail="Associated promo data not found.")

        return {
            "decrypted_data": decrypted,
            "promoter_name": promoter.get("profile_name"),
            "promoter_handle": promoter.get("handle"),
            "restaurant_name": restaurant.get("profile_name"),
            "original_post_caption": post.get("caption"),
            "offer_description": promo_info.get("offer_description"),
            "expiry_date": promo_info.get("expiry_date")
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid or corrupt promo code.")


@api_router.put("/users/{user_id}")
async def update_user(user_id: str, update: UserUpdate, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    update_dict = {k: v for k, v in update.dict().items() if v is not None}
    if update_dict:
        db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_dict})
    
    return {"message": "User updated successfully"}

@api_router.put("/users/{user_id}/location")
async def update_location(user_id: str, location: LocationUpdate, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.get("user_type") != "Restaurant":
        raise HTTPException(status_code=400, detail="Only restaurants can set locations")
    
    # Update restaurant_details with location
    restaurant_details = user.get("restaurant_details", {})
    restaurant_details["location"] = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "address": location.address,
        "place_name": location.place_name
    }
    
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"restaurant_details": restaurant_details}}
    )
    
    return {"message": "Location updated successfully", "location": restaurant_details["location"]}

@api_router.post("/users/{user_id}/follow")
async def follow_user(user_id: str, current_user: Dict = Depends(get_current_user)):
    follower_id = current_user["user_id"]
    if follower_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")

    target = db.users.find_one({"_id": ObjectId(user_id)})
    if not target:
        raise HTTPException(status_code=404, detail="User to follow not found")

    # Add to following list
    db.users.update_one(
        {"_id": ObjectId(follower_id)},
        {"$addToSet": {"following": user_id}}
    )

    # Add to followers list
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$addToSet": {"followers": follower_id}}
    )

    return {"message": "Followed successfully"}

@api_router.post("/users/{user_id}/unfollow")
async def unfollow_user(user_id: str, current_user: Dict = Depends(get_current_user)):
    follower_id = current_user["user_id"]
    if follower_id == user_id:
        raise HTTPException(status_code=400, detail="Cannot unfollow yourself")

    target = db.users.find_one({"_id": ObjectId(user_id)})
    if not target:
        raise HTTPException(status_code=404, detail="User to unfollow not found")

    # Remove from following list
    db.users.update_one(
        {"_id": ObjectId(follower_id)},
        {"$pull": {"following": user_id}}
    )

    # Remove from followers list
    db.users.update_one(
        {"_id": ObjectId(user_id)},
        {"$pull": {"followers": follower_id}}
    )

    return {"message": "Unfollowed successfully"}

@api_router.get("/users/{user_id}/posts")
async def get_user_posts(user_id: str, skip: int = 0, limit: int = 20):
    posts = db.posts.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    for post in posts:
        post["_id"] = str(post["_id"])
    
    return posts

# Promotion Endpoints
@api_router.get("/restaurants/{restaurant_id}/promo_requests")
async def get_promo_requests(restaurant_id: str, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != restaurant_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    posts = db.posts.find({
        "restaurant_tagged_id": restaurant_id,
        "is_promotion_request": True,
        "promotion_status": "Pending"
    }).to_list(100)
    
    # Enrich with user data
    for post in posts:
        post["_id"] = str(post["_id"])
        user = db.users.find_one({"_id": ObjectId(post["user_id"])})
        if user:
            post["user"] = {
                "_id": str(user["_id"]),
                "profile_name": user["profile_name"],
                "handle": user["handle"],
                "avatar_base64": user.get("avatar_base64")
            }
    
    return posts

@api_router.post("/restaurants/{restaurant_id}/promo_requests/{post_id}/approve")
async def approve_promo(restaurant_id: str, post_id: str, promo: PromoApprove, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != restaurant_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    post = db.posts.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Encrypt promo code
    encrypted_code = encrypt_promo_code(
        promo.promo_code_plain_text,
        post["user_id"],
        restaurant_id,
        post_id
    )
    
    # Generate QR code from encrypted string
    qr_base64 = generate_qr_code_base64(encrypted_code)
    
    # Create promo code entry with QR
    promo_dict = {
        "code_encrypted": encrypted_code,
        "qr_base64": qr_base64,
        "promoter_foodie_id": post["user_id"],
        "restaurant_id": restaurant_id,
        "post_id": post_id,
        "offer_description": promo.offer_description,
        "expiry_date": promo.expiry_date,
        "redemptions": [],
        "created_at": datetime.utcnow()
    }
    
    result = db.promocodes.insert_one(promo_dict)
    promo_code_id = str(result.inserted_id)
    
    # Update post
    db.posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$set": {
            "promotion_status": "Approved",
            "promo_code_id": promo_code_id,
            "updated_at": datetime.utcnow()
        }}
    )
    
    return {"message": "Promo approved", "encrypted_code": encrypted_code, "qr_base64": qr_base64}

@api_router.post("/restaurants/{restaurant_id}/promo_requests/{post_id}/reject")
async def reject_promo(restaurant_id: str, post_id: str, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != restaurant_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    db.posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$set": {"promotion_status": "Rejected", "updated_at": datetime.utcnow()}}
    )
    
    return {"message": "Promo rejected"}
@api_router.post("/restaurants/{restaurant_id}/redeem_promo")
async def redeem_promo(restaurant_id: str, redemption: PromoRedeem, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != restaurant_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Decrypt code
    decrypted = decrypt_promo_code(redemption.promo_code_encrypted)
    
    # Verify restaurant
    if decrypted["restaurant_id"] != restaurant_id:
        raise HTTPException(status_code=400, detail="Invalid promo code for this restaurant")

    # --- START: MODIFICATION ---
    # Find the redeemer by their handle (usertag) instead of ID
    redeemer_handle = redemption.redeemer_user_id.strip()
    print("Redeemer handle:", redeemer_handle)


    redeemer_user = db.users.find_one({"handle": redeemer_handle})
    if not redeemer_user:
        raise HTTPException(status_code=404, detail=f"Redeemer with handle '@{redeemer_handle}' not found.")
    
    redeemer_actual_id = str(redeemer_user["_id"])
    # --- END: MODIFICATION ---
    
    # Find promo code
    promo = db.promocodes.find_one({"post_id": decrypted["post_id"]})
    if not promo:
        raise HTTPException(status_code=404, detail="Promo code not found")
    
    # Check expiry
    if promo.get("expiry_date"):
        expiry = datetime.fromisoformat(promo["expiry_date"])
        if datetime.utcnow() > expiry:
            raise HTTPException(status_code=400, detail="Promo code expired")
    
    # Add redemption using the actual ID found from the handle
    redemption_obj = {
        "redeemer_user_id": redeemer_actual_id,
        "redeemed_at": datetime.utcnow(),
        "restaurant_confirmation_status": "Confirmed"
    }
    
    db.promocodes.update_one(
        {"_id": promo["_id"]},
        {"$push": {"redemptions": redemption_obj}}
    )
    
    # Update loyalty points for the original promoter (this logic remains the same)
    promoter_id = decrypted["promoter_id"]
    loyalty = db.loyalty_points.find_one({
        "restaurant_id": restaurant_id,
        "foodie_id": promoter_id
    })
    
    if loyalty:
        db.loyalty_points.update_one(
            {"_id": loyalty["_id"]},
            {
                "$inc": {"points": 10},
                "$push": {"transactions": {
                    "amount": 10,
                    "type": "Earned",
                    "source_promo_code_id": str(promo["_id"]),
                    "date": datetime.utcnow()
                }},
                "$set": {"last_updated": datetime.utcnow()}
            }
        )
    else:
        db.loyalty_points.insert_one({
            "restaurant_id": restaurant_id,
            "foodie_id": promoter_id,
            "points": 10,
            "transactions": [{
                "amount": 10,
                "type": "Earned",
                "source_promo_code_id": str(promo["_id"]),
                "date": datetime.utcnow()
            }],
            "last_updated": datetime.utcnow()
        })
    
    return {"message": "Promo redeemed successfully", "points_awarded": 10}

# Loyalty Points Endpoints
@api_router.get("/users/{foodie_id}/loyalty_points")
async def get_loyalty_points(foodie_id: str, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != foodie_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    loyalty_points = db.loyalty_points.find({"foodie_id": foodie_id}).to_list(100)
    
    for lp in loyalty_points:
        lp["_id"] = str(lp["_id"])
        # Get restaurant info
        restaurant = db.users.find_one({"_id": ObjectId(lp["restaurant_id"])})
        if restaurant:
            lp["restaurant"] = {
                "profile_name": restaurant["profile_name"],
                "avatar_base64": restaurant.get("avatar_base64")
            }
    
    return loyalty_points

@api_router.get("/restaurants/{restaurant_id}/loyalty_points")
async def get_restaurant_loyalty_points(restaurant_id: str, current_user: Dict = Depends(get_current_user)):
    if current_user["user_id"] != restaurant_id:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    loyalty_points = db.loyalty_points.find({"restaurant_id": restaurant_id}).to_list(100)
    
    for lp in loyalty_points:
        lp["_id"] = str(lp["_id"])
        # Get foodie info
        foodie = db.users.find_one({"_id": ObjectId(lp["foodie_id"])})
        if foodie:
            lp["foodie"] = {
                "profile_name": foodie["profile_name"],
                "handle": foodie["handle"],
                "avatar_base64": foodie.get("avatar_base64")
            }
    
    return loyalty_points


# Post Endpoints
@api_router.post("/posts")
async def create_post(post: PostCreate, current_user: Dict = Depends(get_current_user)):
    post_dict = post.dict()
    # Normalize and validate base64 image input
    image_b64 = post_dict.get("image_base64")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_base64 is required")
    # Strip possible data URI prefix
    if image_b64.startswith("data:"):
        try:
            image_b64 = image_b64.split(",", 1)[1]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image_base64 format")
    # Validate base64 (reject invalid characters/padding)
    try:
        # validate=True ensures only proper base64 alphabet is accepted
        base64.b64decode(image_b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 is not valid base64")
    post_dict["image_base64"] = image_b64
    post_dict["user_id"] = current_user["user_id"]
    post_dict["post_type"] = "Promotion" if post.is_promotion_request else "Normal"
    post_dict["likes"] = []
    post_dict["comments"] = []
    post_dict["promotion_status"] = "Pending" if post.is_promotion_request else "N/A"
    post_dict["created_at"] = datetime.utcnow()
    post_dict["updated_at"] = datetime.utcnow()
    
    result = db.posts.insert_one(post_dict)
    post_id = str(result.inserted_id)
    # Broadcast to all foodie clients that a new post is available
    try:
        await feed_manager.broadcast_new_post(post_id, post_dict["post_type"])
    except Exception:
        # Do not block post creation on notification failures
        pass
    
    return {"post_id": post_id, "message": "Post created successfully"}
@api_router.get("/posts/feed/trending")
async def get_trending_feed(city: Optional[str] = None, skip: int = 0, limit: int = 20):
    query = {"promotion_status": {"$in": ["N/A", "Approved"]}}
    
    if city:
        query["location.name"] = {"$regex": city, "$options": "i"}
    
    posts = db.posts.find(query).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    # Enrich posts with user data
    for post in posts:
        post["_id"] = str(post["_id"])
        user = db.users.find_one({"_id": ObjectId(post["user_id"])})
        if user:
            post["user"] = {
                "_id": str(user["_id"]),
                "profile_name": user["profile_name"],
                "handle": user["handle"],
                "avatar_base64": user.get("avatar_base64"),
                "user_type": user["user_type"],
                "image_base64": user.get("image_base64")
            }
        
        # Get promo code if approved
        if post.get("promo_code_id"):
            promo = db.promocodes.find_one({"_id": ObjectId(post["promo_code_id"])})
            if promo:
                post["promo_code"] = promo["code_encrypted"]
                post["promo_qr_base64"] = promo.get("qr_base64", "")
                post["offer_description"] = promo["offer_description"]
        
        # --- START: ADD RESTAURANT LOCATION ---
        if post.get("restaurant_tagged_id"):
            restaurant = db.users.find_one({"_id": ObjectId(post["restaurant_tagged_id"])})
            if restaurant and restaurant.get("restaurant_details", {}).get("location"):
                post["restaurant_location"] = restaurant["restaurant_details"]["location"]
                res_name= restaurant.get("profile_name", "Unknown")
                post["res_name"] = res_name
        # --- END: ADD RESTAURANT LOCATION ---
    
    return posts

@api_router.get("/posts/feed/following")
async def get_following_feed(skip: int = 0, limit: int = 20, current_user: Dict = Depends(get_current_user)):
    user = db.users.find_one({"_id": ObjectId(current_user["user_id"])})
    following = user.get("following", [])
    
    posts = db.posts.find(
        {"user_id": {"$in": following}, "promotion_status": {"$in": ["N/A", "Approved"]}}
    ).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)
    
    # Enrich posts with user data
    for post in posts:
        post["_id"] = str(post["_id"])
        user =  db.users.find_one({"_id": ObjectId(post["user_id"])})
        if user:
            post["user"] = {
                "_id": str(user["_id"]),
                "profile_name": user["profile_name"],
                "handle": user["handle"],
                "avatar_base64": user.get("avatar_base64"),
                "user_type": user["user_type"],
                "image_base64": user.get("image_base64")
            }
        
        # Get promo code if approved
        if post.get("promo_code_id"):
            promo = db.promocodes.find_one({"_id": ObjectId(post["promo_code_id"])})
            if promo:
                post["promo_code"] = promo["code_encrypted"]
                post["promo_qr_base64"] = promo.get("qr_base64", "")
                post["offer_description"] = promo["offer_description"]

        # --- START: ADD RESTAURANT LOCATION ---
        if post.get("restaurant_tagged_id"):
            restaurant = db.users.find_one({"_id": ObjectId(post["restaurant_tagged_id"])})
            if restaurant and restaurant.get("restaurant_details", {}).get("location"):
                post["restaurant_location"] = restaurant["restaurant_details"]["location"]
                res_name= restaurant.get("profile_name", "Unknown")
                post["res_name"] = res_name
        # --- END: ADD RESTAURANT LOCATION ---

    return posts


@api_router.get("/posts/{post_id}")
async def get_post(post_id: str):
    post = db.posts.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    post["_id"] = str(post["_id"])
    
    # Get user data
    user = db.users.find_one({"_id": ObjectId(post["user_id"])})
    if user:
        post["user"] = {
            "_id": str(user["_id"]),
            "profile_name": user["profile_name"],
            "handle": user["handle"],
            "avatar_base64": user.get("avatar_base64"),
            "user_type": user["user_type"],
            "image_base64": user.get("image_base64")
        }
    
    # Get promo code if approved
    if post.get("promo_code_id"):
        promo = db.promocodes.find_one({"_id": ObjectId(post["promo_code_id"])})
        if promo:
            post["promo_code"] = promo["code_encrypted"]
            post["promo_qr_base64"] = promo.get("qr_base64", "")
            post["offer_description"] = promo["offer_description"]

    # --- START: ADD RESTAURANT LOCATION ---
    if post.get("restaurant_tagged_id"):
        restaurant = db.users.find_one({"_id": ObjectId(post["restaurant_tagged_id"])})
        if restaurant and restaurant.get("restaurant_details", {}).get("location"):
            post["restaurant_location"] = restaurant["restaurant_details"]["location"]
            res_name= restaurant.get("profile_name", "Unknown")
            post["res_name"] = res_name
    # --- END: ADD RESTAURANT LOCATION ---
            
    
    return post

@api_router.post("/posts/{post_id}/like")
async def like_post(post_id: str, current_user: Dict = Depends(get_current_user)):
    db.posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$addToSet": {"likes": current_user["user_id"]}}
    )
    return {"message": "Post liked"}

@api_router.post("/posts/{post_id}/unlike")
async def unlike_post(post_id: str, current_user: Dict = Depends(get_current_user)):
    db.posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$pull": {"likes": current_user["user_id"]}}
    )
    return {"message": "Post unliked"}

@api_router.post("/posts/{post_id}/comments")
async def add_comment(post_id: str, comment: CommentCreate, current_user: Dict = Depends(get_current_user)):
    comment_obj = {
        "user_id": current_user["user_id"],
        "text": comment.text,
        "created_at": datetime.utcnow()
    }
    
    db.posts.update_one(
        {"_id": ObjectId(post_id)},
        {"$push": {"comments": comment_obj}}
    )
    
    return {"message": "Comment added"}

@api_router.get("/posts/{post_id}/comments")
async def get_comments(post_id: str):
    post = db.posts.find_one({"_id": ObjectId(post_id)})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    comments = post.get("comments", [])
    
    # Enrich with user data
    for comment in comments:
        user = db.users.find_one({"_id": ObjectId(comment["user_id"])})
        if user:
            comment["user"] = {
                "profile_name": user["profile_name"],
                "handle": user["handle"],
                "avatar_base64": user.get("avatar_base64")
            }
    
    return comments


# New endpoint to fetch posts that tag a restaurant
@api_router.get("/restaurants/{restaurant_id}/tagged_posts")
async def get_tagged_posts(restaurant_id: str, skip: int = 0, limit: int = 50):
    posts = db.posts.find({
        "restaurant_tagged_id": restaurant_id,
        "promotion_status": {"$in": ["N/A", "Approved"]},
    }).sort("created_at", -1).skip(skip).limit(limit).to_list(limit)

    for post in posts:
        post["_id"] = str(post["_id"])
    return posts

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# WebSocket endpoint for realtime feed notifications
@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket, token: str = Query(None)):
    # Simple token verification using existing JWT to associate user
    try:
        if not token:
            await websocket.close(code=4401)
            return
        payload = verify_token(token)
        user_id = payload.get("user_id")
        user = db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            await websocket.close(code=4403)
            return
        # Only Foodies need realtime feed of new posts
        await feed_manager.connect(user_id, websocket)
    except Exception:
        await websocket.close(code=4401)
        return

    try:
        while True:
            # Keep connection alive; server is push-only, but we can receive pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        feed_manager.disconnect(user_id)
