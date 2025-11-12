import json
import pandas as pd


class MCC_CODE:
    """Load MCC Code Data"""
    def __init__(self):
        self.data = None

    def load_json(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            self.data = data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")

    def get_category_name_by_code(self, mcc_code: str) -> str:
        """Return MCC code category name by code"""
        try:
            if self.data['category_name'][mcc_code]:
                return self.data['category_name'][mcc_code]
        except :
            return "Category Not Found"
        

    def get_category_by_range(self, mcc_code: int) -> str:
        """return Category range and Name by checking mcc_code present in Range"""
        for col in self.data['category_range']:
            if mcc_code >= col['min'] and mcc_code <= col['max']:
                return col['category']
        return "No Category Found"




class TXN:
    """load Transations form csv file."""
    def __init__(self):
        self.data = None

    def load_csv(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except:
            print(f"Something bad happed!")

    def bind_mcc_categories(self, mcc_instance):
        """Bind MCC categories to the transaction data"""
        if self.data is None or mcc_instance.data is None:
            print("Error: Data not loaded. Please load both transaction and MCC data first.")
            return

        # Add category column using exact MCC code lookup
        self.data['mcc_category_name'] = self.data['mccCode'].astype(str).map(mcc_instance.get_category_name_by_code)

        # Add category column using range lookup for missing categories
        self.data['mcc_category_range'] = pd.to_numeric(self.data['mccCode'], errors='coerce').map(mcc_instance.get_category_by_range)

        # Fill missing categories from name lookup with range lookup
        self.data['mcc_category'] = self.data['mcc_category_name'].fillna(self.data['mcc_category_range'])

        # Convert TransactionDate to datetime
        self.data['TransactionDate'] = pd.to_datetime(self.data['TransactionDate'], format='%d-%m-%Y', errors='coerce')

        # Add additional time-based columns for analysis
        self.data['year'] = self.data['TransactionDate'].dt.year
        self.data['month'] = self.data['TransactionDate'].dt.month
        self.data['day'] = self.data['TransactionDate'].dt.day
        self.data['day_of_week'] = self.data['TransactionDate'].dt.dayofweek  # Monday=0 to Sunday=6
        self.data['quarter'] = self.data['TransactionDate'].dt.quarter
