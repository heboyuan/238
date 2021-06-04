import random
import matplotlib.pyplot as plt

# incubation period of covid
HIDDEN_TIME = 20
# COVID mutate chance (# in one chance)
MUTATE_PROB = 1/200000
# infection rate (percentage) at different location
# household infection rate reference: https://www.cdc.gov/mmwr/volumes/69/wr/mm6944e1.htm
# office infection rate reference: https://www.cdc.gov/coronavirus/2019-ncov/php/community-mitigation/non-healthcare-work-settings.html
# gym infection rate reference: https://www.advisory.com/en/daily-briefing/2021/03/01/gym-infections
INFECTION_RATE = {"home": 0.53/100, "office": 0.25/100, "gym": 0.68/100}
# maximum possible age
MAX_AGE = 100
# vaccine development cycle
MAX_VACCINE_DEVELOPMENT_TIME = 100

# total death rate by age referenced by
# https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/COVID-19/COVID-19-Cases-by-Age-Group.aspx
death_rate = dict()
# calculate onetime probalility by (1-x)^(heal_time) = 1-total_death_rate
death_prob = dict()
for age in range(MAX_AGE):
    if age < 5:
        death_rate[age] = 0
        death_prob[age] = 0
    elif age <= 17:
        death_rate[age] = 0
        death_prob[age] = 0
    elif age <= 34:
        death_rate[age] = 0.014
        death_prob[age] = 0.00003
    elif age <= 49:
        death_rate[age] = 0.053
        death_prob[age] = 0.00011
    elif age <= 59:
        death_rate[age] = 0.106
        death_prob[age] = 0.00023
    elif age <= 64:
        death_rate[age] = 0.092
        death_prob[age] = 0.0002
    elif age <= 69:
        death_rate[age] = 0.106
        death_prob[age] = 0.00023
    elif age <= 74:
        death_rate[age] = 0.116
        death_prob[age] = 0.000256
    else:
        death_rate[age] = 0.392
        death_prob[age] = 0.001



class Person:
    def __init__(self, gender, age, initial_location, actions, covid = -1):
        self.location = initial_location
        initial_location.add(self)

        self.alive = True
        self.age = age
        # 1 for male 0 for female
        self.gender = gender
        self.vaccine = None
        self.covid = covid
        self.infection_time = -1
        self.antibody = -1
        self.actions = actions

        # recover time from 
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7229949/
        if self.gender:
            if self.age < 20:
                self.recovery_time = 327
            elif self.age < 30:
                self.recovery_time = 327
            elif self.age < 40:
                self.recovery_time = 347
            elif self.age < 50:
                self.recovery_time = 355
            elif self.age < 60:
                self.recovery_time = 355
            else:
                self.recovery_time = 353
        else:
            if self.age < 20:
                self.recovery_time = 318
            elif self.age < 30:
                self.recovery_time = 335
            elif self.age < 40:
                self.recovery_time = 340
            elif self.age < 50:
                self.recovery_time = 354
            elif self.age < 60:
                self.recovery_time = 340
            else:
                self.recovery_time = 336
        
    
    def act(self, current_time):
        global closed_location_type, gathering_size_limit, testing_delay, vaccine_version, vaccination_age
        
        # -3: covid mutated, -2: just died, -1: have covid, 0: healthy, 1: recovered from covid, 2: got vaccine
        return_value = 0
        
        # covid development
        if self.covid >= 0:
            # healing
            self.infection_time += 1
            # fully heal
            if self.infection_time == self.recovery_time:
                self.antibody = self.covid
                self.covid = -1
                self.infection_time = 0
                return_value = 1
            # die with possibility related to age
            elif random.random() <= death_prob[self.age]:
                self.alive = False
                self.location.remove(self)
                return -2
            # COVID mutate
            elif not random.random() < MUTATE_PROB:
               self.covid += 1
               return_value = -3
            else:
                return_value = -1

        # check current location
        if self.location.type in closed_location_type:
            # -2 in aciton is where people go when they have no place to go
            self.location.remove(self)
            self.location = self.actions[-2]
            self.location.add(self)

        # go to next location
        #   no covid or haven't test for covid
        if self.infection_time < HIDDEN_TIME + testing_delay or self.covid < 0 :
            if current_time in self.actions:
                next_location = self.actions[current_time]
                self.location.remove(self)
                if next_location.type in closed_location_type or next_location.people_count >= gathering_size_limit:
                    # -2 in aciton is where people go when they have no place to go
                    self.location = self.actions[-2]
                else:
                    self.location = self.actions[current_time]
                self.location.add(self)
            if random.random() > vaccination_wellingness and self.age > vaccination_age and self.covid < 0:
                self.antibody = vaccine_version
                return_value = 2

        #   tested for covid, so need to go home
        elif self.location != self.actions[-1]:
            # -1 in aciton is where people go when they are sick
            self.location.remove(self)
            self.location = self.actions[-1]
            self.location.add(self)

        return return_value
                
    
    def get_covid(self, infection_rate, covid):
        if random.random() < infection_rate and self.covid < covid and self.antibody < covid:
            self.infection_time = 0
            self.covid = covid

class Location:
    def __init__(self, location_type):
        self.type = location_type         
        self.people = set()
        self.covid = -1
        self.people_count = 0
    
    def infect(self):
        global mask_mandate
        for person in self.people:
             if person.covid > self.covid:
                 self.covid = person.covid
        infection_rate = INFECTION_RATE[self.type]
        # mask mandate reduce infection rate by 70-80 percent
        # data from https://jamanetwork.com/journals/jama/fullarticle/2776536
        if mask_mandate:
            infection_rate *= 0.25
        for person in self.people:
            person.get_covid(infection_rate, self.covid)
    
    def remove(self, person):
        self.people.remove(person)
        self.people_count -= 1
    
    def add(self, person):
        self.people.add(person)
        self.people_count += 1

# agent actions
# global since same for everyone
closed_location_type = set()
mask_mandate = 0
gathering_size_limit = float("inf")
testing_delay = 0
vaccine_version = -1
vaccine_dev_time = 0
vaccination_age = MAX_AGE
vaccination_wellingness = 0.0

class Environment:

    def __init__(self, locations, people):
        self.locations = []
        self.people = []
        self.time = 0
        # assume we have covid to start with
        self.covid_version = 0

        locations_construction = {
            "office": [],
            "gym": [],
            "home": []
        }

        for location_type, location_count in locations:
            for _ in range(location_count):
                loc = Location(location_type)
                locations_construction[location_type].append(loc)
                self.locations.append(loc)
        people_home_index = 0
        for male_percentage, age, action_source, people_count in people:
            for _ in range(people_count):
                if people_home_index < len(locations_construction["home"]):
                    home = locations_construction["home"][people_home_index]
                    people_home_index += 1
                else:
                    home = random.choice(locations_construction["home"])
                gender = random.randint(0, 99) < male_percentage
                
                acts = {-2:home, -1:home}
                for time, location_type in action_source:
                    if location_type == "home":
                        acts[time] = home
                    else:
                        acts[time] = random.choice(locations_construction[location_type])
                
                if random.randint(0, 100):
                    self.people.append(Person(gender, age, home, acts))
                else:
                    self.people.append(Person(gender, age, home, acts, 0))
        
        # stats features
        self.vaccine_counts = []
        self.covid_counts = []
        self.total_covid_counts = []
        self.healthy_counts = []
        self.accu_death_counts = []
        self.accu_recover_counts = []

    def step(self, actions):
        # development code ==================
        # print(actions)
        # development code ==================
        global closed_location_type, mask_mandate, vaccine_version, vaccine_dev_time

        # perform different actions
        #  close locations
        if actions[0]:
            closed_location_type.add("gym")
        else:
            closed_location_type.discard("gym")
        if actions[1]:
            closed_location_type.add("office")
        else:
            closed_location_type.discard("office")
        # mask mandate
        mask_mandate = actions[2]
        # TODO: actions[3] limit gathering size. how to set limit? boolean? number in range?
        # TODO: actions[4] testing delay. how to set test delay? boolean? number in range?
        # TODO: action[5] set the age of people allowed to get vaccine. how to set test delay? boolean? number in range?
        # TODO: action[6] set vaccination wellingness. how to set test delay? boolean? number in range?
        
        # stats variables
        mutation_indicator = False
        covid_count = [0 for _ in range(self.covid_version + 1)]
        healthy_count = 0
        vaccine_count = 0
        accu_death_count = self.accu_death_counts[-1] if self.accu_death_counts else 0
        accu_recover_count = self.accu_recover_counts[-1] if self.accu_recover_counts else 0
        # run the environment
        for person in self.people:
            if not person.alive:
                continue
            action_return = person.act(self.time%24)
            if action_return == 2:
                vaccine_count += 1
                healthy_count += 1
            elif action_return == 1:
                accu_recover_count += 1
                healthy_count += 1
            elif action_return == -1:
                covid_count[person.covid] += 1
            elif action_return == -2:
                accu_death_count += 1
            elif action_return == -3:
                if self.covid_version < person.covid:
                    covid_count.append(1)
                    self.covid_version = person.covid
                else:
                    covid_count[person.covid] += 1
            else:
                healthy_count += 1
        self.vaccine_counts.append(vaccine_count)
        self.accu_death_counts.append(accu_death_count)
        self.accu_recover_counts.append(accu_recover_count)
        self.healthy_counts.append(healthy_count)
        self.total_covid_counts.append(sum(covid_count))

        while self.covid_version + 1 > len(self.covid_counts):
            self.covid_counts.append([0 for _ in range(self.time)])
        for version, count in enumerate(covid_count):
            self.covid_counts[version].append(count)    

        # print("covid version", self.covid_version)
        # print("covid", sum(covid_count), "death", accu_death_count, "recover", accu_recover_count, "healthy", healthy_count, "vaccine", vaccine_count)


        for location in self.locations:
            if location.type not in closed_location_type:
                location.infect()
        if vaccine_version < self.covid_version:
            vaccine_dev_time = random.random()*MAX_VACCINE_DEVELOPMENT_TIME
            
        if vaccine_dev_time > 0:
            vaccine_dev_time -= 1
        else:
            vaccine_version = self.covid_version

        self.time += 1
    
    def print_time(self):
        print("day", self.time//24, "time", self.time%24)

    def print_people(self, person_id):
        person = self.people[person_id]
        print("person", person_id, "at " + person.location.type, "covid version", person.covid, "antibody version", person.antibody, "infection time", person.infection_time)
    
    def print_location(self, location_id):
        location = self.locations[location_id]
        print("location", location_id, "type " + location.type, "people count", location.people_count)
    
    def plot(self):
        time_line = [t for t in range(self.time)]
        for covid_version, covid_count in enumerate(self.covid_counts):
            plt.plot(time_line, covid_count, label = "covid version " + str(covid_version))
        plt.plot(time_line, self.accu_death_counts, label = "covid death")
        plt.plot(time_line, self.total_covid_counts, label = "all covid version count")
        # plt.plot(time_line, self.accu_recover_counts, label = "covid recovery")
        # plt.plot(time_line, self.healthy_counts, label = "healthy people")
        plt.legend()
        plt.savefig('test.png')

    
    # def print_general_info(self):
    #     healthy_count = 0
    #     death_count = 0
    #     covid_count = 0
    #     personal_antibody_count = 0
    #     global_antibody_count = 0
    #     for person in self.people:
    #         if not person.alive:
    #             death_count += 1
    #             continue
    #         if person.covid >= 0:
    #             covid_count += 1
    #         else:
    #             healthy_count += 1
    #         if person.antibody >= person.covid:
    #             personal_antibody_count += 1
    #         if person.antibody >= self.covid_version:
    #             global_antibody_count += 1

    #     print("covid", covid_count, "death", death_count, "healthy", healthy_count)
    #     print("personal effective antibody count", personal_antibody_count)
    #     print("global effective antibody count",global_antibody_count)
    #     print("")



ACTION_TEMPLATE_1 = [(9, "office"), (17, "gym"), (20, "home")]
ACTION_TEMPLATE_2 = [(9, "office"), (17, "home")]

def main():
    locations = [("gym", 50), ("office", 500), ("home", 50000)]
    people = [(70, 80, ACTION_TEMPLATE_1, 50000), (40, 80, ACTION_TEMPLATE_2, 50000)] 
    env = Environment(locations, people)

    while True:
        # env.print_time()
        actions = [0,0,0]
        env.step(actions)
        # actions = list(map(int, list(input("Enter your actions:"))))
        # if not (env.time-1) % 240 and input("Hit Enter to continue...\n"):
        #         env.plot()
        if env.time == 200:
            return
        
        # env.print_general_info()

if __name__ == "__main__":
    main()




        

