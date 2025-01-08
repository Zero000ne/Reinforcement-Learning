import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# 2D 환경 정의 (격자에서 목표를 찾는 환경)
class GridEnv:
    def __init__(self, size=5):
        self.size = size # 격자 크기
        self.agent_position = [0, 0] # 에이전트 시작 위치
        self.goal_position = [size-1, size-1] # 목표 위치 (우측 하단)
        self.done = False
        self.state = self.reset() #초기 상태
        
    def reset(self):
        self.agent_position = [0, 0]  # 에이전트 초기 위치
        self.done = False
        return np.array(self.agent_position)

    def step(self, action):
        # 에이전트의 현재 위치
        x, y = self.agent_position

        # 행동에 따라 위치 업데이트 (상, 하, 좌, 우)
        if action == 0:  # 위
            x = max(0, x - 1)
        elif action == 1:  # 아래
            x = min(self.size - 1, x + 1)
        elif action == 2:  # 왼쪽
            y = max(0, y - 1)
        elif action == 3:  # 오른쪽
            y = min(self.size - 1, y + 1)

        self.agent_position = [x, y]

        # 목표에 도달하면 보상 1, 벽에 부딪히면 -1
        if self.agent_position == self.goal_position:
            self.done = True
            reward = 1
        elif self.agent_position == [0, 0] and action == 0:  # 예시: 왼쪽 위로 이동하면 벽에 부딪힘
            reward = -1
        else:
            reward = 0  # 기본 보상 (중간 상태)

        return np.array(self.agent_position), reward, self.done



class QNet(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(QNet,self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)  # 은닉층2
        self.fc3 = nn.Linear(64, output_dim)  # 출력층

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self,env,gamma=0.99, epsilon=0.1, batch_size=32, memory_size=1000, learning_rate=0.001):
        self.env = env
        self.gamma = gamma  # 할인율
        self.epsilon = epsilon  # 탐색(exploration) 비율
        self.batch_size = batch_size  # 배치 크기
        self.memory = deque(maxlen=memory_size)  # 경험 리플레이 메모리
        self.model = QNet(2,4) # 상태: 2D, 행동: 4개 (상, 하, 좌, 우)
        self.target_model = QNet(2, 4)  # 타겟 네트워크
        self.optimizer = optim.Adam(self.model.parameters(),lr=learning_rate)

    def act(self, state):
        if random.uniform(0,1) < self.epsilon:
            return random.choice([0, 1, 2, 3]) #랜덤 행동
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0) # 상태 텐서로 변환
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # 가장 큰 Q-값을 가진 행동 반환

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # 경험 저장

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype= torch.float32)
        next_states = torch.tensor(next_states, dtype= torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        #Q-값 계산
        current_q_values = self.model(states).gather(1, torch.tensor(actions).view(-1, 1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # 손실 계산
        loss = torch.mean((current_q_values - target_q_values.view(-1, 1)) ** 2)

        # 경사 하강법으로 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        def update_target_model(self):
            self.target_model.load_state_dict(self.model.state_dict())  # 타겟 네트워크 업데이트



# DQN 에이전트 학습
env = GridEnv(size=5)  # 5x5 크기의 격자 환경
agent = DQNAgent(env)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state) # 행동 선택
        next_state, reward, done = env.step(action)  # 환경에서 1스텝 실행
        agent.remember(state, action, reward, next_state, done)  # 경험 저장
        agent.replay()  # 경험 리플레이 학습
        state = next_state  # 상태 갱신
        total_reward += reward

    # 타겟 네트워크 업데이트
    agent.update_target_model()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")
