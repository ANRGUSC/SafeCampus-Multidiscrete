import math
from scipy.stats import binom

def estimate_infected_students(current_infected, allowed_per_course, community_risk, total_students):

    infected_students = []
    total_population = total_students
    for i in range(len(allowed_per_course)):
        const_1 = 0.005  # reduce this to a smaller value
        const_2 = 0.01  # reduce this value to be very small 0.01, 0.02
        infected = int(((const_1 * current_infected[i]) * (allowed_per_course[i])) + (
                (const_2 * community_risk) * allowed_per_course[i] ** 2))

        infected = min(infected, allowed_per_course[i])

        infected_students.append(infected)

    infected_students = list(map(int, list(map(round, infected_students))))
    return infected_students

def estimate_infected_students_sir(current_infected, allowed_per_course, community_risk, total_students):

    infected_students = []
    total_students = total_students
    # Iterate over each course
    for i in range(len(allowed_per_course)):
        # Constants for infection rates inside and outside the course
        const_1 = 0.001  # reduce this to a smaller value
        const_2 = 0.01  # reduce this value to be very small 0.01, 0.02

        # Recovery rate for infected students
        recovery_rate = 1.0

        # Calculate the number of susceptible students in the course
        susceptible = max(0, total_students - current_infected[i])

        # Calculate the number of susceptible students in the course
        # susceptible = max(0, total_students - current_infected[i])

        # Estimate new infections within the course
        new_infected_inside = int(
            (const_1 * current_infected[i]) * (susceptible / 100) * susceptible * allowed_per_course[i])

        # Estimate new infections from the community
        new_infected_outside = int((const_2 * community_risk * allowed_per_course[i]) * susceptible)

        # Estimate recovered students
        recovered = max(int(recovery_rate * current_infected[i]), 0)

        # Calculate the total number of new infections
        total_infected = new_infected_inside + new_infected_outside
        # Calculate the total number of infected students after accounting for recoveries

        infected = min(current_infected[i] + int(total_infected) - recovered, allowed_per_course[i])

        # Append the result to the list of infected students
        infected_students.append(infected)

    # Round and convert the infected students to integers
    infected_students = list(map(int, list(map(round, infected_students))))

    return infected_students


# Constants
ROOM_AREA = 669 # default: 869  # [SQFT] # for 100 students
ROOM_ACH = 12.12 / 3600  # [1/s]
ROOM_HEIGHT = 2.7  # m
HVAC_EFFICIENCY = 0.8
ACTIVE_INFECTED_TIME = 0.2
MAX_DURATION = 12 * 60  # minutes
BREATH_RATE = 2 * 10 ** -4  # Breathing rate of the occupants
ACTIVE_INFECTED_EMISSION = 40
PASSIVE_INFECTION_EMISSION = 1
D0_values = [500, 1000, 2000]  # Different values of D0 for testing
D0 = 1000
VACCINATION_EFFECT = 0.85
DOSE_VACCINATION_RATIO = 0.25
TRANSMISSION_VACCINATION_RATIO = 0.25
community_risk_values = [i / 10 for i in range(11)]  # Values from 0 to 1
allowed_values = [0, 50, 100]  # Values for allowed

def calculate_indoor_infection_prob(room_capacity: int, initial_infection_prob: float):
    occupancy_density = room_capacity / (ROOM_AREA * 0.092903)
    dose_one_person = (
            occupancy_density * BREATH_RATE /
            (ROOM_HEIGHT * HVAC_EFFICIENCY * ROOM_ACH) *
            (ACTIVE_INFECTED_TIME * ACTIVE_INFECTED_EMISSION +
             (1 - ACTIVE_INFECTED_TIME) * PASSIVE_INFECTION_EMISSION) * MAX_DURATION
    )
    total_transmission_prob = 0
    for i in range(0, room_capacity):
        infection_prob = binom.pmf(i, room_capacity, initial_infection_prob)
        dose_total = i * dose_one_person
        # if dose_total > 0:
        transmission_prob = 1 - math.exp(-dose_total / D0)
        # else:
        #     transmission_prob = 0
        total_transmission_prob += infection_prob * transmission_prob

    return total_transmission_prob

def get_infected_students(current_infected_students: list, allowed_students_per_course: list, community_risk: float, total_students: int):
    infected_students = []
    for n, f in enumerate(allowed_students_per_course):
        susceptible = max(0, total_students - current_infected_students[n])
        asymptomatic_ratio = 0.5
        recovery_rate = 0.1
        room_capacity = allowed_students_per_course[n]
        initial_infection_prob = (current_infected_students[n] / total_students) *\
                                 susceptible/total_students

        infected_prob = calculate_indoor_infection_prob(room_capacity, initial_infection_prob)
        # Ensure they are valid
        if math.isnan(infected_prob):
            infected_prob = 0  # or another default value
        if math.isnan(allowed_students_per_course[n]):
            allowed_students_per_course[n] = 0  # or another default value
        total_indoor_infected_allowed = int(infected_prob * allowed_students_per_course[n])

        total_infected_allowed_outdoor = int(community_risk * allowed_students_per_course[n])
        total_infected_allowed = min(total_indoor_infected_allowed + total_infected_allowed_outdoor,
                                    room_capacity)
        recovered = max(int(recovery_rate * current_infected_students[n]), 0)
        infected_students.append(
            int(round(min((current_infected_students[n] + total_infected_allowed - recovered), room_capacity))))

    return infected_students

