import random

# how long does it take to heal from covid
HEAL_TIME = 480
# incubation period of covid
HIDDEN_TIME = 20
# COVID mutate chance 
MUTATE_PROB = 10
# infection rate (percentage) at different location
# household infection rate reference: https://www.cdc.gov/mmwr/volumes/69/wr/mm6944e1.htm
# office infection rate reference: https://www.cdc.gov/coronavirus/2019-ncov/php/community-mitigation/non-healthcare-work-settings.html
# gym infection rate reference: https://www.advisory.com/en/daily-briefing/2021/03/01/gym-infections
INFECTION_RATE = {"home": 53, "office": 25, "gym": 68}
# id counter for location and people
location_id, person_id = 0, 0
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
        global person_id
        self.id = person_id
        person_id += 1

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
        if not self.alive:
            return
        
        # covid development
        if self.covid >= 0:
            # COVID mutate
            if random.randrange(100) < MUTATE_PROB:
               self.covid += 1 
            
            # healing
            self.infection_time += 1
            # die with possibility related to age
            if random.random() <= death_prob[self.age]:
                self.alive = False
            # fully heal
            if self.infection_time == self.recovery_time:
                self.antibody = self.covid
                self.covid = -1
                self.infection_time = 0

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
            if random.randrange(100) > vaccination_wellingness and self.age > vaccination_age:
                self.antibody = vaccine_version

        #   tested for covid, so need to go home
        elif self.location != self.actions[-1]:
            # -1 in aciton is where people go when they are sick
            self.location.remove(self)
            self.location = self.actions[-1]
            self.location.add(self)
                
    
    def get_covid(self, infection_rate, covid):
        if random.randrange(100) < infection_rate and self.covid < covid and self.antibody < covid:
            self.infection_time = 0
            self.covid = covid

class Location:
    def __init__(self, location_type):
        global location_id
        self.id = location_id
        location_id += 1

        self.type = location_type         
        self.people = set()
        self.covid = -1
        self.people_count = 0
    
    def infect(self):
        global mask_mandate
        if self.people:
            # TODO: died people can still spread COVID?
            self.covid = max(self.covid, max(self.people, key=lambda p:p.covid).covid)
        infection_rate = INFECTION_RATE[self.type]
        # mask mandate reduce infection rate by 70-80 percent
        # data from https://jamanetwork.com/journals/jama/fullarticle/2776536
        if mask_mandate:
            infection_rate *= 0.25
        for person in self.people:
            person.get_covid(infection_rate, self.covid)
        return self.covid
    
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
vaccination_wellingness = 0

class Environment:

    def __init__(self):
        self.locations = []
        self.people = []
        self.time = 0

        # development code ================== 
        home0 = Location("home")
        home1 = Location("home")
        office0 = Location("office")
        gym0 = Location("gym")
        self.locations = [home0, home1, office0, gym0]
        self.people = [ Person(1, 35, home0, {-2:home0, -1:home0, 2:office0, 17:gym0, 19:home0}, 6),
                        Person(0, 35, home1, {-2:home1, -1:home1, 2:office0, 17:gym0, 19:home1})]
        # development code ==================

    def step(self, actions):
        # development code ==================
        print(actions)
        # development code ==================
        global closed_location_type, mask_mandate, vaccine_version

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
        
        # run the environment
        covid_version = -1
        for person in self.people:
            person.act(self.time%24)
        for location in self.locations:
            if location.type not in closed_location_type:
                covid_version = max(covid_version, location.infect())
        if vaccine_version != covid_version:
            vaccine_dev_time = random.randrange(MAX_VACCINE_DEVELOPMENT_TIME)
            
        if vaccine_dev_time:
            vaccine_dev_time -= 1
        else:
            vaccine_version = covid_version

        self.time += 1
    
    def print(self):
        person0 = self.people[0]
        person1 = self.people[1]
        print("time", self.time)
        print("person 0 at " + person0.location.type, "covid version", person0.covid, "antibody version", person0.antibody, "infection time", person0.infection_time)
        print("person 1 at " + person1.location.type, "covid version", person1.covid, "antibody version", person1.antibody, "infection time", person1.infection_time)


def main():
    env = Environment()

    while True:
        actions = list(map(int, list(input("Enter your actions:"))))
        env.step(actions)
        env.print()

if __name__ == "__main__":
    main()




        

