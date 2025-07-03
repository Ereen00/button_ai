import tkinter as tk
import torch
import torch.nn as nn
import torch.optim as optim

#-------------------------------------------------------------

# Neural network definition
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model and optimization settings
input_size = 2  # Changed to 2 to accommodate row and column
hidden_size = 50  # Hidden layer size
output_size = 20  # Changed to 20 to accommodate 20 buttons
model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.01)  # Corrected 'RMSPROP' to 'RMSprop'

#-------------------------------------------------------------

# Data storage
data = []  # Stores previous (row, column) inputs
labels = []  # Stores previous button IDs
accuracy_rates = []  # Stores accuracy rates for each round
predictions = []  # Stores model's predictions for each round
actuals = []  # Stores actual user inputs for each round

def on_button_click(button_id, row, column):
    global data, labels, accuracy_rates, predictions, actuals

    # Update data and target
    data.append([row, column]) # Row and column information
    labels.append(button_id) # Stores previous button IDs

    if len(data) > 15: # Keep only the last 15 entries
        data.pop(0)
        labels.pop(0)

    print(f"Clicked Button: {button_id + 1}, Position: ({row}, {column})")
    print(f"Data: {data}")

    # Change button color
    if len(data) >= 5:
        buttons[labels[-5]].config(bg='SystemButtonFace')  # Reset color of 5 clicks ago button
    buttons[button_id].config(bg='lightblue')  # Change color of current button

    # Perform model prediction and training
    if len(data) == 15: # Train only if there is enough data
        # Create input and target tensors
        input_tensor = torch.tensor(data, dtype=torch.float32) # Input tensor
        target_tensor = torch.tensor(labels, dtype=torch.long) # Target for the last button

        # Model prediction
        output = model(input_tensor) # Model output

        # Predicted button
        _, predicted = torch.max(output, 1)  # Get predicted class for each output

        correct_count = 0
        for i in range(len(predicted)):
            if predicted[i].item() == labels[i]:
                correct_count += 1

        accuracy = correct_count / len(predicted)
        accuracy_rates.append(accuracy)

        predictions.append(predicted.tolist())
        actuals.append(labels)

        print(f"AI Predictions: {predicted.tolist()}, Actuals: {labels}")
        print(f"Accuracy: {accuracy}")

        # Save accuracy rates and predictions to file
        with open("accuracy_rates.txt", "w") as f:
            for i in range(len(accuracy_rates)):
                f.write(f"Round {i+1}:\n")
                f.write(f"Accuracy: {accuracy_rates[i]}\n")
                f.write(f"Predictions: {predictions[i]}\n")
                f.write(f"Actuals: {actuals[i]}\n\n")

        # Train the model
        optimizer.zero_grad()
        loss = criterion(output, target_tensor)  # Calculate loss
        loss.backward() # Backpropagation
        optimizer.step() # Update weights

        print(f"Training Loss: {loss.item()}")

#-------------------------------------------------------------

# Tkinter interface
root = tk.Tk()
root.title("AI Button Predictor")

buttons = []
rows, columns = 4, 5  # 4 rows, 5 columns
for i in range(rows):
    for j in range(columns):
        btn_id = i * columns + j
        btn = tk.Button(root, text=f"Button {btn_id + 1}",
                        command=lambda btn_id=btn_id, i=i, j=j: on_button_click(btn_id, i, j))
        btn.grid(row=i, column=j, padx=5, pady=5)
        buttons.append(btn)

root.mainloop()