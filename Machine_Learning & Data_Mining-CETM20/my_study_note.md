You're correct—hyperparameters can be grouped by their impact on preprocessing, model architecture, optimization, and regularization. Here’s a summary of how each group helps with **overfitting** or **underfitting**, and when to use them:

---

### 1. **Preprocessing Hyperparameters**
- **Standardization/Normalization:**  
  - *Purpose:* Ensures features are on similar scales, helping optimization.
  - *Helps with:* Underfitting (by making training easier), but not directly with overfitting.
  - *Use when:* Data features have different scales or distributions.

---

### 2. **Model Architecture Hyperparameters**
- **Type of Model (Dense, CNN, RNN, Transformer):**  
  - *Purpose:* Matches model to data type (images, text, etc.).
  - *Helps with:* Underfitting (if model is too simple), overfitting (if model is too complex).
  - *Use when:* Data type changes or model is not learning well.

- **Number of Layers/Neurons:**  
  - *Purpose:* Controls model capacity.
  - *Helps with:* Underfitting (increase layers/neurons), overfitting (decrease layers/neurons).
  - *Use when:* Model is too simple (underfit) or too complex (overfit).

- **Activation Function:**  
  - *Purpose:* Adds non-linearity.
  - *Helps with:* Underfitting (choose better activation), not directly with overfitting.
  - *Use when:* Model can't capture complex patterns.

---

### 3. **Optimization Hyperparameters**
- **Learning Rate:**  
  - *Purpose:* Controls step size in optimization.
  - *Helps with:* Underfitting (increase if learning is slow), overfitting (decrease if model jumps to bad minima).
  - *Use when:* Loss stagnates or oscillates.

- **Batch Size:**  
  - *Purpose:* Number of samples per gradient update.
  - *Helps with:* Underfitting (smaller batch can help generalization), overfitting (larger batch can stabilize).
  - *Use when:* Training is unstable or slow.

- **Epochs:**  
  - *Purpose:* Number of passes through data.
  - *Helps with:* Underfitting (increase epochs), overfitting (decrease epochs).
  - *Use when:* Model hasn't learned enough or starts to overfit.

- **Loss Function/Metric:**  
  - *Purpose:* Guides optimization.
  - *Helps with:* Underfitting (choose appropriate loss), not directly with overfitting.
  - *Use when:* Task changes (classification vs regression).

---

### 4. **Regularization Hyperparameters**
- **Dropout:**  
  - *Purpose:* Randomly drops neurons during training.
  - *Helps with:* Overfitting.
  - *Use when:* Validation loss diverges from training loss.

- **L1/L2 Regularization:**  
  - *Purpose:* Penalizes large weights.
  - *Helps with:* Overfitting.
  - *Use when:* Model memorizes training data.

- **Early Stopping:**  
  - *Purpose:* Stops training when validation loss increases.
  - *Helps with:* Overfitting.
  - *Use when:* Validation loss starts to rise.

---

## **Mapping Hyperparameters to Problems**

| Problem         | Hyperparameter Lever(s) to Use                |
|-----------------|-----------------------------------------------|
| Overfitting     | Regularization (Dropout, L1/L2), Early Stopping, Reduce Model Size, Reduce Epochs |
| Underfitting    | Increase Model Size, More Layers/Neurons, Change Activation, Increase Epochs, Adjust Learning Rate |
| Poor Generalization | Regularization, Data Augmentation, Better Preprocessing, Early Stopping |
| Training Loss Stagnates | Change Learning Rate, More Epochs, Better Preprocessing |
| Validation Loss Diverges | Regularization, Reduce Model Complexity, Early Stopping |

---

### **Diagram: Hyperparameter Levers**

