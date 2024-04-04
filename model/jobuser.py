""" database dependencies to support sqliteDB examples """

from datetime import date

import json

from __init__ import app, db
from sqlalchemy.exc import IntegrityError

''' Tutorial: https://www.sqlalchemy.org/library.html#tutorials, try to get into Python shell and follow along '''



# Define the job class to manage actions in the 'jobs' table
# -- Object Relational Mapping (ORM) is the key concept of SQLAlchemy
# -- a.) db.Model is like an inner layer of the onion in ORM
# -- b.) job represents data we want to store, something that is built on db.Model
# -- c.) SQLAlchemy ORM is layer on top of SQLAlchemy Core, then SQLAlchemy engine, SQL
class JobUser(db.Model):
    __tablename__ = 'jobsusers'  # table title is plural, class title is singular

    # Define the job schema with "vars" from object
    id = db.Column(db.Integer, primary_key=True)
    userid = db.Column(db.Integer, db.ForeignKey('aidansusers.id'))
    jobid = db.Column(db.Integer, db.ForeignKey('jobs.id'))
    
    dateApplied = db.Column(db.Date)


    
    # constructor of a job object, initializes the instance variables within object (self)
    def __init__(self, jobid, userid, dateApplied=date.today()):
        self.jobid = jobid
        self.userid = userid
        self.dateApplied = dateApplied
       
    def create(self):
        try:
            # creates a person object from User(db.Model) class, passes initializers
            db.session.add(self)  # add prepares to persist person object to Users table
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
            "jobid": self.jobid,
            "userid": self.userid,
            "dateApplied": self.dateApplied,
   
            
            # "posts": [post.read() for post in self.posts]
        }
    

# Builds working data for testing
def initJobsUsers():
    with app.app_context():
        """Create database and tables"""
        db.create_all()
        """Tester data for table"""
        jobs = [
            JobUser(jobid=1, userid=2, dateApplied=date(1847, 2, 11)),
            JobUser(jobid=2, userid=3, dateApplied=date(2024, 4, 12))
        ]
        for job in jobs:
            try:
                job.create()
            except IntegrityError:
                '''fails with bad or duplicate data'''
                db.session.remove()
                print(f"Records exist, duplicate title, or error: {job.title}")

