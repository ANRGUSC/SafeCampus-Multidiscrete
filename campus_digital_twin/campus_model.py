import random
random.seed(100)
class CampusModel:
    def __init__(self, num_courses=1, students_per_course=None, max_weeks=16, initial_infection_rate=0.2):
        self.num_courses = num_courses

        # Handle varying students per course
        if students_per_course is None:
            self.students_per_course = [random.choice([10, 100]) for _ in range(num_courses)]
        elif isinstance(students_per_course, int):
            self.students_per_course = [students_per_course] * num_courses
        elif isinstance(students_per_course, list) and len(students_per_course) == num_courses:
            self.students_per_course = students_per_course
        else:
            raise ValueError("Invalid students_per_course input")

        self.total_students = sum(self.students_per_course)
        self.max_weeks = max_weeks

        # Handle initial infection rate
        if isinstance(initial_infection_rate, (int, float)):
            self.initial_infection_rate = [initial_infection_rate] * num_courses
        elif isinstance(initial_infection_rate, list) and len(initial_infection_rate) == num_courses:
            self.initial_infection_rate = initial_infection_rate
        else:
            raise ValueError("Invalid initial_infection_rate input")

        # Calculate initial number of infected students per course
        self.initial_infection = [int(rate * students) for rate, students in
                                  zip(self.initial_infection_rate, self.students_per_course)]

    def number_of_students_per_course(self):
        return self.students_per_course

    def get_max_weeks(self):
        return self.max_weeks

    def get_initial_infection(self):
        return self.initial_infection

