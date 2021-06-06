import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ==========================================================================================================
#                                          START environment START                                         |
# ==========================================================================================================

# incubation period of covid
HIDDEN_TIME = 20
# COVID mutate chance
MUTATE_PROB = 0
# infection rate (percentage) at different location
# household infection rate reference: https://www.cdc.gov/mmwr/volumes/69/wr/mm6944e1.htm
# office infection rate reference: https://www.cdc.gov/coronavirus/2019-ncov/php/community-mitigation/non-healthcare-work-settings.html
# gym infection rate reference: https://www.advisory.com/en/daily-briefing/2021/03/01/gym-infections
INFECTION_RATE = {"home": 0.53/100, "office": 0.25/100, "gym": 0.68/100}
# maximum possible age
MAX_AGE = 100
# vaccine development cycle
MAX_VACCINE_DEVELOPMENT_TIME = 200

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
        initial_location.people_count += 1

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
        global closed_location_type, gathering_size_limit, testing_delay, vaccine_version, vaccination_age, mask_mandate
        
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
                self.location.people_count -= 1
                return -2
            # COVID mutate
            elif random.random() < MUTATE_PROB:
               self.covid += 1
               return_value = -3
            else:
                return_value = -1

        # check current location
        if self.location.type in closed_location_type:
            # -2 in aciton is where people go when they have no place to go
            self.location.people_count -= 1
            self.location = self.actions[-2]
            self.location.people_count += 1

        # go to next location
        #   no covid or haven't test for covid
        if self.infection_time < HIDDEN_TIME + testing_delay or self.covid < 0 :
            if current_time in self.actions:
                next_location = self.actions[current_time]
                self.location.people_count -= 1
                if next_location.type in closed_location_type or next_location.people_count >= gathering_size_limit:
                    # -2 in aciton is where people go when they have no place to go
                    self.location = self.actions[-2]
                else:
                    self.location = self.actions[current_time]
                self.location.people_count += 1
            if random.random() < vaccination_wellingness and self.age > vaccination_age and self.covid < 0 and self.antibody < vaccine_version and vaccine_version != -1:
                self.antibody = vaccine_version
                return_value = 2

        #   tested for covid, so need to go home
        elif self.location != self.actions[-1]:
            # -1 in aciton is where people go when they are sick
            self.location.people_count -= 1
            self.location = self.actions[-1]
            self.location.people_count += 1

        if self.location.covid < self.covid:
            self.location.covid = self.covid
        return return_value       
    
    def get_covid(self):
        covid = self.location.covid
        infection_rate = self.location.base_infection_rate
        if mask_mandate:
            infection_rate = infection_rate*0.25
        if self.covid < covid and self.antibody < covid and random.random() < infection_rate:
            self.infection_time = 0
            self.covid = covid
            

class Location:
    def __init__(self, location_type):
        self.type = location_type
        self.base_infection_rate = INFECTION_RATE[location_type]
        self.covid = -1
        self.people_count = 0

# agent actions
# global since same for everyone
closed_location_type = set()
mask_mandate = 0
gathering_size_limit = float("inf")
testing_delay = 0
vaccine_version = -1
vaccine_dev_version = -1
vaccine_dev_time = 0
vaccination_age = MAX_AGE
vaccination_wellingness = 0.0

def reset_global_for_environment():
    global closed_location_type, mask_mandate, gathering_size_limit, testing_delay, vaccine_version,\
            vaccine_dev_version, vaccine_dev_time, vaccination_age, vaccination_wellingness
    closed_location_type = set()
    mask_mandate = 0
    gathering_size_limit = float("inf")
    testing_delay = 0
    vaccine_version = -1
    vaccine_dev_version = -1
    vaccine_dev_time = 0
    vaccination_age = MAX_AGE
    vaccination_wellingness = 1


# configure constant
INITIAL_COVID_PERCENTAGE = 0.01

OBSERVATION_COVID_VERSION = 10
OBSERVATION_VACCINE_VERSION = 10
N_ACTIONS = 7

class Environment:

    def __init__(self, locations, people, debug=False):
        self.locations = []
        self.people = []
        self.time = 0
        # assume we have covid to start with
        self.covid_version = 0
        self.debug = debug

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
                
                if random.random() < INITIAL_COVID_PERCENTAGE:
                    self.people.append(Person(gender, age, home, acts, 0))
                else:
                    self.people.append(Person(gender, age, home, acts))
        
        # stats features
        self.vaccine_counts = []
        self.covid_counts = []
        self.total_covid_counts = []
        self.healthy_counts = []
        self.accu_death_counts = []
        self.accu_recover_counts = []

    def step(self, action):
        global closed_location_type, mask_mandate, vaccine_version, vaccine_dev_version, vaccine_dev_time, vaccination_age

        # perform different actions
        #  close locations
        if action == 1:
            closed_location_type.add("gym")
        elif action == 2 :
            closed_location_type.discard("gym")
        elif action == 3:
            closed_location_type.add("office")
        elif action == 4:
            closed_location_type.discard("office")
        elif action == 5:
            mask_mandate = 1
        elif action == 6:
            mask_mandate = 0
        elif action == 7:
            vaccination_age -= 5
        # TODO: actions[3] limit gathering size. how to set limit? boolean? number in range?
        # TODO: actions[4] testing delay. how to set test delay? boolean? number in range?
        # TODO: action[5] set the age of people allowed to get vaccine. how to set test delay? boolean? number in range?
        # TODO: action[6] set vaccination wellingness. how to set test delay? boolean? number in range?
        
        # stats variables
        mutation_indicator = False
        covid_count = [0 for _ in range(self.covid_version + 1)]
        vaccine_count = self.vaccine_counts[-1][-1] if self.vaccine_counts else 0
        healthy_count = 0
        accu_death_count = self.accu_death_counts[-1] if self.accu_death_counts else 0
        accu_recover_count = self.accu_recover_counts[-1] if self.accu_recover_counts else 0
        # run the environment
        for location in self.locations:
            location.covid = -1
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
        self.accu_death_counts.append(accu_death_count)
        self.accu_recover_counts.append(accu_recover_count)
        self.healthy_counts.append(healthy_count)
        self.total_covid_counts.append(sum(covid_count))

        while self.covid_version + 1 > len(self.covid_counts):
            self.covid_counts.append([float('nan') for _ in range(self.time)])
        for version, count in enumerate(covid_count):
            self.covid_counts[version].append(count)

        if self.vaccine_counts:
            self.vaccine_counts[-1].append(vaccine_count)
        for vaccine_log in self.vaccine_counts[:-1]:
            vaccine_log.append(vaccine_log[-1])
        if vaccine_dev_version < self.covid_version:
            vaccine_dev_version = self.covid_version
            vaccine_dev_time = random.random()*MAX_VACCINE_DEVELOPMENT_TIME
        if vaccine_dev_time > 0:
            vaccine_dev_time -= 1
        else:
            vaccine_version = vaccine_dev_version
            self.vaccine_counts += [[0 for _ in range(self.time)] for _ in range(vaccine_version - len(self.vaccine_counts) + 1)] 
        
        
        for person in self.people:
            person.get_covid()
        
        if self.debug:
            print(self.time)
            print("covid version", self.covid_version, "vaccine version", vaccine_version)
            print("covid", self.total_covid_counts[-1], "death", accu_death_count, "recover",\
                    accu_recover_count, "healthy", healthy_count, "latest vaccine count", vaccine_count )
            print("")
        self.time += 1   

        # observation
        covid_version_case = [0]*OBSERVATION_COVID_VERSION
        if len(covid_count) > OBSERVATION_COVID_VERSION:
            covid_version_case = covid_count[-OBSERVATION_COVID_VERSION:]
        else:
            covid_version_case = covid_count[:] + [0]*(OBSERVATION_COVID_VERSION - len(covid_count))
        covid_version = self.covid_version

        
        vaccine_version_dose = []
        if len(self.vaccine_counts) > OBSERVATION_VACCINE_VERSION:
            vaccine_version_dose = [vaccine_log[-1] for vaccine_log in self.vaccine_counts[-OBSERVATION_VACCINE_VERSION:]]
        elif not len(self.vaccine_counts):
            vaccine_version_dose = [0]*OBSERVATION_VACCINE_VERSION
        else:
            vaccine_version_dose = [vaccine_log[-1] for vaccine_log in self.vaccine_counts] + [0]*(OBSERVATION_VACCINE_VERSION - len(self.vaccine_counts))

        observation = covid_version_case + [covid_version] + vaccine_version_dose + [vaccine_version] + [accu_death_count] + [accu_recover_count] + [healthy_count]
        observation = torch.FloatTensor(observation).unsqueeze(0)
        # reward
        reward = 0
        if len(self.total_covid_counts) == 1:
            reward = -self.total_covid_counts[-1]
        elif len(self.total_covid_counts) > 1:
            reward = self.total_covid_counts[-2] - self.total_covid_counts[-1]
            if reward > 0:
                reward = 0

        # done
        done = self.total_covid_counts[-1] == 0

        return observation, reward, done
    
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
        plt.close()
        for covid_version, covid_count in enumerate(self.covid_counts):
            plt.plot(time_line, covid_count, label = "covid version " + str(covid_version))
        plt.plot(time_line, self.accu_death_counts, label = "covid death")
        plt.plot(time_line, self.total_covid_counts, label = "all covid version count")
        # plt.plot(time_line, self.accu_recover_counts, label = "covid recovery")
        # plt.plot(time_line, self.healthy_counts, label = "healthy people")
        plt.legend()
        plt.savefig('test.png')
# ==========================================================================================================
#                                            END environment END                                           |
# ==========================================================================================================

# ==========================================================================================================
#                                              START DQN START                                             |
# ==========================================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 20000
TARGET_UPDATE = 10

observation, _, _ = Environment([], []).step(0)
reset_global_for_environment()
input_size = list(observation.size())[1]
policy_net = DQN(input_size, N_ACTIONS).to(device)
target_net = DQN(input_size, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(100000)

def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(N_ACTIONS)]], device=device, dtype=torch.long)

episode_durations = []
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('test_cuda.png')

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# ==========================================================================================================
#                                                END DQN END                                               |
# ==========================================================================================================

ACTION_TEMPLATE_1 = [(9, "office"), (17, "gym"), (20, "home")]
ACTION_TEMPLATE_2 = [(9, "office"), (17, "home")]

def main():
    locations = [("gym", 10), ("office", 50), ("home", 4000)]
    people = [(70, 20, ACTION_TEMPLATE_1, 5000), (40, 20, ACTION_TEMPLATE_2, 5000)] 
    
    #============================
    # training code             |
    #============================
    steps_done = 0
    num_episodes = 100
    for i_episode in range(num_episodes):
        env = Environment(locations, people)
        reset_global_for_environment()
        state, _, _ = env.step(0)
        done = False
        accumulate_reward = 0
        while not done:
            action = select_action(state, steps_done)
            # print(action, steps_done, i_episode)
            steps_done += 1
            next_state, reward, done = env.step(action.item())
            accumulate_reward += reward
            reward = torch.tensor([reward], device=device)
            if done:
                next_state = None
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
        episode_durations.append(accumulate_reward)
        plot_durations()
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        print(i_episode, steps_done)

    #============================
    # debug code                |
    #============================
    # acc_reward = 0
    # env = Environment(locations, people, True)
    # while True:
    #     # input("Hit Enter to continue...\n")
    #     # if env.time and env.time % 120 == 0 and input("Hit Enter to continue...\n"):
    #     #         env.plot()
    #     #         continue
    #     # actions = list(map(int, list(input("Enter your actions:"))))
    #     _, reward, done = env.step(7)
    #     acc_reward+=reward
    #     print(reward)
    #     if(done):
    #         break
    # print(acc_reward)

if __name__ == "__main__":
    main()




        

