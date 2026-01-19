from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
import os

class connection:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        with open(config_path) as f:
            self.keys = json.load(f)
        DATABASE_URL=self.keys.get("DATA_BASE","")
        print(DATABASE_URL)
        self.engine=create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        from model import Base
        Base.metadata.create_all(self.engine)
        print("Tables created successfully!")

if __name__ == "__main__":
    connect=connection()
    connect.create_tables()