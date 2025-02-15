import numpy as np
import pandas as pd

class ARCE:
    def __init__(self, initial_vigilance=0.8, learning_rate=0.1):
        self.vigilance = initial_vigilance
        self.learning_rate = learning_rate
        self.categories = []  # List to store category prototypes
        self.context_weights = {} # Dictionary to store context weights for each category

    def _contextual_embedding(self, input_data, context):
      """Embeds context with the input.  Simple concatenation for now."""
      # More sophisticated methods could be used here (e.g., learned embeddings)
      if isinstance(context, dict):
        context_vector = np.array(list(context.values())) # Convert context to vector
      elif isinstance(context, pd.Series):
        context_vector = context.values
      else:
        context_vector = np.array(context) # Assume it's already a vector or list

      return np.concatenate((input_data, context_vector))

    def _resonance_check(self, embedded_input, category_prototype):
        """Checks if the input resonates with a category."""
        distance = np.linalg.norm(embedded_input - category_prototype)
        match_ratio = 1 - (distance / (np.linalg.norm(embedded_input) + 1e-7)) # Avoid division by zero
        return match_ratio >= self.vigilance

    def _update_category(self, embedded_input, category_prototype):
      """Updates category prototype based on the input."""
      category_prototype += self.learning_rate * (embedded_input - category_prototype)
      return category_prototype

    def train(self, input_data, context):
        embedded_input = self._contextual_embedding(input_data, context)
        best_match_category = None
        best_match_ratio = -1

        for i, category_prototype in enumerate(self.categories):
            match_ratio = self._resonance_check(embedded_input, category_prototype)
            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_match_category = i

        if best_match_category is not None:
            # Resonance occurred, update the category
            self.categories[best_match_category] = self._update_category(embedded_input, self.categories[best_match_category])
            # Update context weights (example: increase weight for relevant context features)
            for context_feature in context:
                if context_feature in self.context_weights[best_match_category]:
                    self.context_weights[best_match_category][context_feature] += self.learning_rate
                else:
                    self.context_weights[best_match_category][context_feature] = self.learning_rate


        else:
            # No resonance, create a new category
            self.categories.append(embedded_input)
            self.context_weights[len(self.categories) - 1] = {}  # Initialize context weights for the new category
            for context_feature in context: # Initialize weights for all features in the context
                self.context_weights[len(self.categories) - 1][context_feature] = self.learning_rate/10 # Small initial weight


    def predict(self, input_data, context):
      embedded_input = self._contextual_embedding(input_data, context)
      best_match_category = None
      best_match_ratio = -1

      for i, category_prototype in enumerate(self.categories):
          match_ratio = self._resonance_check(embedded_input, category_prototype)
          if match_ratio > best_match_ratio:
              best_match_ratio = match_ratio
              best_match_category = i
      return best_match_category # Returns the index of the best matching category or None if no match


# Example usage (Illustrative):

# Sample Context
contexts = [
    {"time": 9, "location": "office"}, # Morning at office
    {"time": 12, "location": "restaurant"}, # Lunch at restaurant
    {"time": 15, "location": "office"}, # Afternoon at office
    {"time": 19, "location": "home"}, # Evening at home
    {"time": 21, "location": "home"}, # Late evening at home
]

# Sample Input Data (e.g., sensor readings, user activity)
input_data = [
    [10, 20],  # Input 1
    [12, 22],  # Input 2
    [11, 19],  # Input 3
    [5, 8], # Input 4
    [6, 9]  # Input 5
]

arce_net = ARCE()

# Training
for i in range(len(input_data)):
  arce_net.train(input_data[i], contexts[i])

# Prediction
print("Prediction for [11, 21] in context {'time': 16, 'location': 'office'}:", arce_net.predict([11, 21], {'time': 16, 'location': 'office'}))
print("Prediction for [5, 7] in context {'time': 20, 'location': 'home'}:", arce_net.predict([5, 7], {'time': 20, 'location': 'home'}))


# Print Category Prototypes and Context Weights
print("Category Prototypes:", arce_net.categories)
print("Context Weights:", arce_net.context_weights)