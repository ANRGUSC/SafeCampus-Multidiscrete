import random
import csv
import logging

class CampusModel:
    def __init__(self, num_courses=1, students_per_course=100, max_weeks=16, initial_infection_rate=0.2, read_weeks_from_csv=False, csv_path=None):
        self.num_courses = num_courses

        # Override students per course and initial infection based on fixed values
        self.students_per_course = [students_per_course] * num_courses
        self.initial_infection_rate = [initial_infection_rate] * num_courses
        self.initial_infection = [20] * num_courses  # Ensure initial infection is always 20

        self.total_students = sum(self.students_per_course)

        # Initialize community risk values list
        self.community_risk_values = []

        # Handle reading max weeks from CSV
        if read_weeks_from_csv and csv_path is not None:
            self._read_weeks_from_csv(csv_path)
        else:
            self.max_weeks = max_weeks
            logging.info(f"Max weeks set from default: {self.max_weeks}")

    def _read_weeks_from_csv(self, csv_path):
        try:
            with open(csv_path, mode='r') as file:
                reader = csv.DictReader(file)
                self.community_risk_values = [float(row['Risk-Level']) for row in reader]
                self.max_weeks = len(self.community_risk_values)
                logging.info(f"Max weeks set from CSV: {self.max_weeks}")
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            raise ValueError(f"Error reading CSV file: {e}")

    def number_of_students_per_course(self):
        return self.students_per_course

    def get_max_weeks(self):
        return self.max_weeks

    def get_initial_infection(self):
        return self.initial_infection
