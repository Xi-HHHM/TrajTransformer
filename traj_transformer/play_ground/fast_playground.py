import numpy as np
from transformers import AutoProcessor
import matplotlib.pyplot as plt

# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

# Tokenize & decode action chunks (we use dummy data here)
action_data = np.random.rand(1, 50, 14)    # [batch, timesteps, action_dim]
tokens = tokenizer(action_data)              # tokens = list[int]
decoded_actions = tokenizer.decode(tokens)

print(tokens[0])
print(len(tokens[0]))

# plt.plot(action_data[0, :, 0], 'r')
# plt.plot(decoded_actions[0, :, 0], 'b')
#
# plt.show()