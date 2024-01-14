# ERA-SESSION24 -  Reinforcement Learning

## Car Game
- Perform Experiments on different maps for running the car and figuring out the roads.

### Model Architecture

```python
class Network(nn.Module):
   
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30, nb_action)
   
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
```
### Results:

![image](https://github.com/Navyabhat03/ERA-V1-Session-24/assets/60884505/afa5cba3-df9d-4de9-8843-5cd2a7d22739)

![image](https://github.com/Navyabhat03/ERA-V1-Session-24/assets/60884505/7276943f-9a24-4ee7-a574-302ce63400d4)


## Reinforcement_UCBerkeley 
- Perform Experiments on puzzle game to achive reward as soon as possible.

### Results:

![reinforcement_result](https://github.com/Navyabhat03/ERA-V1-Session-24/assets/60884505/2932894b-0c09-42a3-8cd0-66a6e126f52e)


