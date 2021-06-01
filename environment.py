import random

# how long does it take to heal from covid
HEAL_TIME = 480
# incubation period of covid
HIDDEN_TIME = 20
# COVID mutate chance 
MUTATE_PROB = 10
# infection rate (percentage) at different location
INFECTION_RATE = {"home": 100, "office": 40, "gym": 70}
# id counter for location and people
location_id, person_id = 0, 0
# maximum possible age
MAX_AGE = 100

# death rate by age referenced by
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
        
    
    def act(self, current_time, closed_location_type):
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
        #   no covid or covid is hidden
        if self.infection_time < HIDDEN_TIME or self.covid < 0 :
            if current_time in self.actions:
                self.location.remove(self)
                if self.actions[current_time].type in closed_location_type:
                    # -2 in aciton is where people go when they have no place to go
                    self.location = self.actions[-2]
                else:
                    self.location = self.actions[current_time]
                self.location.add(self)
        #   covid start to show symptom 
        elif self.infection_time == HIDDEN_TIME:
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
        self.infection_rate = INFECTION_RATE[location_type]
        self.people = set()
        self.covid = -1
    
    def infect(self):
        if self.people:
            # TODO: died people can still spread COVID?
            self.covid = max(self.covid, max(self.people, key=lambda p:p.covid).covid)
        for person in self.people:
            person.get_covid(self.infection_rate, self.covid)
    
    def remove(self, person):
        self.people.remove(person)
    
    def add(self, person):
        self.people.add(person)


class Environment:

    def __init__(self):
        self.locations = []
        self.closed_location_type = set()
        self.people = []
        self.time = 0

        # development code ================== 
        home0 = Location("home")
        home1 = Location("home")
        office0 = Location("office")
        gym0 = Location("gym")
        self.locations = [home0, home1, office0, gym0]
        self.people = [Person(1, 35, home0, {-2:home0, -1:home0, 9:office0, 17:gym0, 19:home0}, 6), Person(0, 35, home1, {-2:home1, -1:home1, 9:office0, 17:gym0, 19:home1})]
        # development code ==================

    def step(self, actions):
        # development code ==================
        print(actions)
        # development code ==================

        # perform different actions
        if actions[0]:
            self.closed_location_type.add("gym")
        else:
            self.closed_location_type.discard("gym")
        if actions[1]:
            self.closed_location_type.add("office")
        else:
            self.closed_location_type.discard("office")
        
        # run the environment
        for person in self.people:
            person.act(self.time%24, self.closed_location_type)
        for location in self.locations:
            if location.type not in self.closed_location_type:
                location.infect()
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




        

