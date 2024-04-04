""" database dependencies to support sqliteDB examples """
from random import randrange
from datetime import date
import os, base64
import json
from .jobuser import JobUser

from __init__ import app, db
from sqlalchemy.exc import IntegrityError

''' Tutorial: https://www.sqlalchemy.org/library.html#tutorials, try to get into Python shell and follow along '''



# Define the job class to manage actions in the 'jobs' table
# -- Object Relational Mapping (ORM) is the key concept of SQLAlchemy
# -- a.) db.Model is like an inner layer of the onion in ORM
# -- b.) job represents data we want to store, something that is built on db.Model
# -- c.) SQLAlchemy ORM is layer on top of SQLAlchemy Core, then SQLAlchemy engine, SQL
class Job(db.Model):
    __tablename__ = 'jobs'  # table title is plural, class title is singular

    # Define the job schema with "vars" from object
    id = db.Column(db.Integer, primary_key=True)
    _title = db.Column(db.String(255), unique=False, nullable=False)
    _description = db.Column(db.String(255), unique=False, nullable=False)
    _field = db.Column(db.String(255), unique=False, nullable=False)
    _location = db.Column(db.String(255), unique=False, nullable=False)
    _qualification = db.Column(db.String(255), unique=False, nullable=False)
    _pay = db.Column(db.Integer, unique=False, nullable=False)
    _jobpostee = db.Column(db.Integer, db.ForeignKey('aidansusers.id'))
    
    jobs = db.relationship('JobUser', backref='jobs', uselist=True, lazy='dynamic')
    applications = db.relationship('Application', backref='jobs', uselist=True, lazy='dynamic')
    
    # constructor of a job object, initializes the instance variables within object (self)
    def __init__(self, title, description, field="IT", location="On-site", qualification="Masters", pay=1000, jobpostee=1):
        self._title = title    # variables with self prefix become part of the object, 
        self._description = description
        self._field = field
        self._location = location
        self._qualification = qualification
        self._pay = pay
        self._jobpostee = jobpostee

    # a title getter method, extracts title from object
    @property
    def title(self):
        return self._title
    
    # a setter function, allows title to be updated after initial object creation
    @title.setter
    def title(self, title):
        self._title = title
        

   
    # a getter method, extracts email from object
    @property
    def description(self):
        return self._description
    
    # a setter function, allows title to be updated after initial object creation
    @description.setter
    def description(self, description):
        self._description = description
    
    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, field):
        self._field = field
        
    # location property is returned as string, to avoid unfriendly outcomes
    @property
    def location(self):
        return self._location

    # location should be have verification for type date
    @location.setter
    def location(self, location):
        self._location = location

    
    # output content using str(object) in human readable form, uses getter
    # output content using json dumps, this is ready for API response
    def __str__(self):
        return json.dumps(self.read())
   
    # qualification is used to store python dictionary data 
    @property
    def qualification(self):
        return self._qualification
    
    @qualification.setter
    def qualification(self, qualification):
        self._qualification = qualification
        
    @property
    def pay(self):
        return self._pay
    
    # a setter function, allows title to be updated after initial object creation
    @pay.setter
    def pay(self, pay):
        self._pay = pay
        
    @property
    def jobpostee(self):
        return self._jobpostee

    @jobpostee.setter
    def jobpostee(self, jobpostee):
        self._jobpostee = jobpostee

    # CRUD create/add a new record to the table
    # returns self or None on error
    def create(self):
        try:
            # creates a person object from job(db.Model) class, passes initializers
            db.session.add(self)  # add prepares to persist person object to jobs table
            db.session.commit()  # SqlAlchemy "unit of work pattern" requires a manual commit
            return self
        except IntegrityError:
            db.session.remove()
            return None

    # CRUD read converts self to dictionary
    # returns dictionary
    def read(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "field": self.field,
            "location": self.location,
            "qualification": self.qualification,
            "pay": self.pay,
            "jobpostee": self.jobpostee
            
            # "posts": [post.read() for post in self.posts]
        }



    # CRUD update: updates job title, field, phone
    # returns self
    def update(self, title="", description="", field="", location="", qualification="", pay=""):
        """only updates values with length"""
        if len(title) > 0:
            self._title = title    # variables with self prefix become part of the object, 
        if len(description) > 0:
            self._description = description
        if len(field) > 0:
            self._field = field
        if len(location) > 0:
            self._location = location
        if len(qualification) > 0:
            self._qualification = qualification
        if pay is not None:
            self._pay = pay
        db.session.commit()
        return self

    # CRUD delete: remove self
    # None
    def delete(self):
        db.session.delete(self)
        db.session.commit()
        return None


"""Database Creation and Testing """


# Builds working data for testing
def initJobs():
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        """Tester data for table"""
        jobs = [
            Job(title='Software Engineer', description='Proficient experience in Java', field="Software", qualification="Mastes", location="Remote", pay=32, jobpostee=1),
            Job(title='Web Developer', description='Proficient experience in Node', field="Web", location="Remote", pay=30),
            Job(title='UX Designer', description='Proficient experience in React', field="Software", location="Remote", pay=25),
            Job(title='IT Technician', description='Proficient experience in computers', field="IT", location="On-site", pay=2)
        ]
        
        for job in jobs:
            try:
                job.create()
            except IntegrityError:
                '''fails with bad or duplicate data'''
                db.session.remove()
                print(f"Records exist, duplicate title, or error: {job.title}")

if __name__ == '__main__':
    initJobs()