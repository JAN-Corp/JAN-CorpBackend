from __init__ import app, db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import os
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError
import pandas as pd


class House(db.Model, UserMixin):
    __tablename__ = 'houses'

    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(256), index=True)
    city = db.Column(db.String(64), index=True, nullable=True)
    state = db.Column(db.String(8), index=True, nullable=True)
    zip = db.Column(db.Integer, index=True, nullable=True)
    latitude = db.Column(db.Float, index=True, nullable=True)
    longitude = db.Column(db.Float, index=True, nullable=True)
    price = db.Column(db.Integer, index=True, nullable=True)
    bathrooms = db.Column(db.Float, index=True, nullable=True)
    bedrooms = db.Column(db.Integer, index=True, nullable=True)
    livingarea = db.Column(db.Integer, index=True, nullable=True)
    homeType = db.Column(db.String(64), index=True, nullable=True)
    priceEstimate = db.Column(db.Integer, index=True, nullable=True)
    rentEstimate = db.Column(db.Integer, index=True, nullable=True)
    imgSRC = db.Column(db.String, index=True, nullable=True)
    favorites = db.relationship('Favorite', backref='House', uselist=True, lazy='dynamic')

    def __init__(self, address, city, state, zip, latitude, longitude, price, bathrooms, bedrooms, livingarea, homeType, priceEstimate, rentEstimate, imgSRC):
        self.address = address
        self.city = city
        self.state = state
        self.zip = zip
        self.latitude = latitude
        self.longitude = longitude
        self.price = price
        self.bathrooms = bathrooms
        self.bedrooms = bedrooms
        self.livingarea = livingarea
        self.homeType = homeType
        self.priceEstimate = priceEstimate
        self.rentEstimate = rentEstimate
        self.imgSRC = imgSRC

    def all_details(self):
        return {
            'id': self.id,
            'address': self.address,
            'city': self.city,
            'state': self.state,
            'zip': self.zip,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'price': self.price,
            'bathrooms': self.bathrooms,
            'bedrooms': self.bedrooms,
            'livingarea': self.livingarea,
            'homeType': self.homeType,
            'priceEstimate': self.priceEstimate,
            'rentEstimate': self.rentEstimate,
            'imgSRC': self.imgSRC
        }

    def few_details(self):
        return {
            'id': self.id,
            'address': self.address,
            'price': self.price,
            'livingarea': self.livingarea,
            'bathrooms': self.bathrooms,
            'bedrooms': self.bedrooms,
            'imgSRC': self.imgSRC
        }


class Favorite(db.Model):
    __tablename__ = 'favorites'
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    house_id = db.Column(db.Integer, db.ForeignKey('houses.id'))

    def __init__(self, account_id, house_id):
        self.account_id = account_id
        self.house_id = house_id


def initHouses():
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        house_count = db.session.query(House).count()
        if house_count > 0:
            return
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        # Specify the file path
        file_path = basedir + "/../static/data/RealEstateData.csv"
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        for index, row in df.iterrows():
            try:
                house = House(
                    address=row['address'] if pd.notna(row['address']) else None,
                    city=row['city'] if pd.notna(row['city']) else None,
                    state=row['state'] if pd.notna(row['state']) else None,
                    zip=row['zipcode'] if pd.notna(row['zipcode']) else None,
                    latitude=row['latitude'] if pd.notna(row['latitude']) else None,
                    longitude=row['longitude'] if pd.notna(row['longitude']) else None,
                    price=row['price'] if pd.notna(row['price']) else None,
                    bathrooms=row['bathrooms'] if pd.notna(row['bathrooms']) else None,
                    bedrooms=row['bedrooms'] if pd.notna(row['bedrooms']) else None,
                    livingarea=row['livingArea'] if pd.notna(row['livingArea']) else None,
                    homeType=row['homeType'] if pd.notna(row['homeType']) else None,
                    priceEstimate=row['PriceEstimate'] if pd.notna(row['PriceEstimate']) else None,
                    rentEstimate=row['RentEstimate'] if pd.notna(row['RentEstimate']) else None,
                    imgSRC=row['imgSrc'] if pd.notna(row['imgSrc']) else None
                )
                db.session.add(house)
                db.session.commit()
            except IntegrityError:
                '''fails with bad or duplicate data'''
                db.session.remove()
                print(f"Records exist, duplicate house, or error: {house.name}")
            except Exception as e_inner:
                print(f"Error adding house at index {index}: {str(e_inner)}")