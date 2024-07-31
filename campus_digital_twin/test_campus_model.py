class CampusModel:
    def __init__(self, num_courses=1, students_per_course=10, max_weeks=15, initial_infection_rate=0.1):
        self.num_courses = num_courses
        self.students_per_course = [students_per_course] * num_courses  # Example: Same number of students for each course
        self.max_weeks = max_weeks
        # Construct the list of initial infection rates for each course
        self.initial_infection_rate = [initial_infection_rate] * num_courses

        # Calculate the initial number of infected students per course based on the infection rate
        self.initial_infection = [int(rate * students) for rate, students in
                                  zip(self.initial_infection_rate, self.students_per_course)]

    def number_of_students_per_course(self):
        # Return the deterministically generated number of students for each course
        return self.students_per_course

    def get_max_weeks(self):
        # Return the number of weeks
        return self.max_weeks

    def get_initial_infection(self):
        # Return the initial number of infected students per course
        return self.initial_infection

    # def initial_community_risk(self):
    #     # Return the deterministically generated community risk values
    #     return self.community_risk_values

    # Other methods can also use deterministic values or logic as needed


# # Example Usage
# if __name__ == "__main__":
#     model = DeterministicCampusModel(num_courses=3)
#     print(model.number_of_students_per_course())  # [30, 30, 30]
#     print(model.initial_community_risk())  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
