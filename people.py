import random

# TODO:
# 1. recovery related to age gender https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7229949/
# 2. death related to age gender..?
# 3. implement hospital?

# how long does it take to heal from covid
HEAL_TIME = 480
# incubation period of covid
HIDDEN_TIME = 100
# COVID mutate chance 
MUTATE_PROB = 10
# infection rate (percentage) at different location
INFECTION_RATE = {"home": 100, "office": 40, "gym": 70}
# id counter for location and people
location_id, person_id = 0, 0

class Person:
    def __init__(self, age, initial_location, actions, covid = -1):
        global person_id
        self.id = person_id
        person_id += 1

        self.location = initial_location
        initial_location.add(self)

        self.alive = True
        self.age = age
        self.vaccine = None
        self.covid = covid
        self.infection_time = -1
        self.antibody = -1
        self.actions = actions
        
    
    def act(self, current_time):
        if self.covid >= 0:
            # COVID mutate
            if random.randrange(100) < MUTATE_PROB:
               self.covid += 1 
            # start healing
            self.infection_time += 1
            # fully healed
            if self.infection_time == HEAL_TIME:
                self.antibody = self.covid
                self.covid = -1
        if self.infection_time < HIDDEN_TIME or self.covid < 0 :
            for time, location in self.actions:
                if current_time == time:
                    self.location.remove(self)
                    self.location = location
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
            self.covid = max(self.covid, max(self.people, key=lambda p:p.covid).covid)
        for person in self.people:
            person.get_covid(self.infection_rate, self.covid)
    
    def remove(self, person):
        self.people.remove(person)
    
    def add(self, person):
        self.people.add(person)


def main():
    home0 = Location("home")
    home1 = Location("home")
    office0 = Location("office")
    gym0 = Location("gym")
    person0 = Person(35, home0, [(9,office0), (17,gym0), (19,home0)], 6)
    person1 = Person(35, home1, [(9,office0), (17,gym0), (19,home1)])

    time = 0
    while True:
        input("Press Enter to continue...")
        person0.act(time)
        person1.act(time)
        home0.infect()
        home1.infect()
        office0.infect()
        gym0.infect()
        time += 1
        time %= 24
        print("time", time)
        print("person 0 at " + person0.location.type, "covid version", person0.covid, "antibody version", person0.antibody, "infection time", person0.infection_time)
        print("person 1 at " + person1.location.type, "covid version", person1.covid, "antibody version", person1.antibody, "infection time", person1.infection_time)

if __name__ == "__main__":
    main()




        

