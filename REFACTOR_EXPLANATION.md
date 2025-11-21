# Train.py Refactoring Explanation

## Overview
We refactored `train.py` from a **functional approach** (360 lines) to a **class-based approach** (~310 lines). While the line count is similar, the new version is much more **reusable** and **maintainable**.

---

## Key Concepts

### What is a Class?
A **class** is a blueprint or template for creating objects. Think of it like a recipe that can be reused.

**Analogy**:
- **Class** = Cookie cutter (the template)
- **Object** = The actual cookie you make with it

```python
# Creating an object from a class
trainer = IsolationForestTrainer(config_path)  # Make a cookie from the cutter
trainer.train()  # Use the cookie (call a method)
```

### What is Inheritance?
**Inheritance** means one class can "inherit" methods and properties from another class.

**Analogy**:
- **Parent class (ModelTrainer)** = Basic car with engine, wheels, steering
- **Child classes** = Sedan, Truck, SUV (all have engine/wheels, but different body styles)

```python
class ModelTrainer:  # Parent
    def train(self):
        # Common training logic

class IsolationForestTrainer(ModelTrainer):  # Child
    # Inherits train() automatically
    # Only needs to define what's different
```

---

## Side-by-Side Comparison

### OLD: Functional Approach (train.py)

```python
# Separate functions for everything
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_scaler(scaler_type):
    if scaler_type == 'standard':
        return StandardScaler()
    # ... more code

def load_feature_data(dataset_config):
    features_path = Path(dataset_config['paths']['features_output_path'])
    # ... more code

def filter_normal_baseline(df, dataset_config):
    # ... lots of code

def select_feature_columns(df, model_config, dataset_config):
    # ... more code

def train_isolation_forest(X_train, hyperparams):
    # ... specific to Isolation Forest

def train_knn_lof(X_train, hyperparams):
    # ... specific to kNN

def train_one_class_svm(X_train, hyperparams):
    # ... specific to SVM

def evaluate_on_training_data(model, X_train, model_type):
    # ... more code

def save_model_and_metadata(...):
    # ... lots of code

def main():
    # Load configs
    model_config = load_config(args.config)
    dataset_config = load_config(dataset_config_path)

    # Load data
    df = load_feature_data(dataset_config)

    # Filter and select
    df_train = filter_normal_baseline(df, dataset_config)
    feature_cols = select_feature_columns(df_train, model_config, dataset_config)

    # Train based on type
    if model_type == 'isolation_forest':
        model = train_isolation_forest(X_train, hyperparams)
    elif model_type == 'knn_lof':
        model = train_knn_lof(X_train, hyperparams)
    elif model_type == 'one_class_svm':
        model = train_one_class_svm(X_train, hyperparams)

    # Evaluate and save
    eval_stats = evaluate_on_training_data(model, X_train, model_type)
    save_model_and_metadata(...)
```

**Problems:**
- Every function needs data passed to it
- Adding a new model = lots of copy-paste
- Have to update if/elif chains
- Hard to reuse in other scripts

---

### NEW: Class-Based Approach (train_refactored.py)

```python
class ModelTrainer:
    """Base class - the template all models follow"""

    def __init__(self, config_path):
        # Load everything once, store it
        self.model_config = self._load_yaml(config_path)
        self.dataset_config = self._load_yaml(...)
        self.hyperparams = self.model_config.get('hyperparameters', {})

    def load_data(self):
        # Load features
        return pd.read_parquet(...)

    def prepare_data(self):
        # All the data prep in one place
        df = self.load_data()
        df_train = self.filter_normal_baseline(df)
        self.feature_cols = self.select_features(df_train)
        # ... scale, handle NaNs, etc.
        return X_train

    def _create_model(self):
        # LEFT BLANK - each model fills this in
        raise NotImplementedError("Subclass must implement")

    def train(self):
        # The main orchestrator
        X_train = self.prepare_data()
        self.model = self._create_model()  # Calls child's version
        self.model.fit(X_train)
        eval_stats = self.evaluate()
        self.save(training_stats)

    def evaluate(self):
        # Evaluation logic
        predictions = self.model.predict(self.X_train)
        # ... calculate stats
        return stats

    def save(self, training_stats):
        # Save model and metadata
        joblib.dump(self.model, model_path)
        # ... save everything

# Each model is tiny now!
class IsolationForestTrainer(ModelTrainer):
    def _create_model(self):
        return IsolationForest(
            n_estimators=self.hyperparams.get('n_estimators', 100),
            contamination=self.hyperparams.get('contamination', 0.1),
            n_jobs=-1
        )

class KNNLOFTrainer(ModelTrainer):
    def _create_model(self):
        return LocalOutlierFactor(
            n_neighbors=self.hyperparams.get('n_neighbors', 20),
            novelty=True,
            n_jobs=-1
        )

# The magic dictionary
MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer
}

def main():
    # Super simple now!
    config = yaml.safe_load(open(args.config))
    model_type = config['model_type']

    trainer = MODEL_TRAINERS[model_type](args.config)
    trainer.train()
```

**Benefits:**
- Data stays with the trainer (no passing around)
- Adding a new model = 10 lines
- No if/elif chains
- Can reuse trainer in other scripts

---

## The Magic: How Inheritance Works

When you write this:
```python
class IsolationForestTrainer(ModelTrainer):
```

The child class **automatically gets** all the parent's methods:
- `__init__()` - setup
- `load_data()` - loading
- `prepare_data()` - preparation
- `train()` - orchestration
- `evaluate()` - evaluation
- `save()` - saving

It **only needs to define** what's unique:
- `_create_model()` - how to create THIS specific model

### Visual Flow

```
trainer = IsolationForestTrainer(config)  # Create object
trainer.train()  # Call train method

What happens inside train():
┌─────────────────────────────────────┐
│ train() [from ModelTrainer base]   │
├─────────────────────────────────────┤
│ 1. X_train = self.prepare_data()   │ ← Uses parent's method
│ 2. self.model = self._create_model()│ ← Uses CHILD's method!
│ 3. self.model.fit(X_train)         │
│ 4. self.evaluate()                 │ ← Uses parent's method
│ 5. self.save()                     │ ← Uses parent's method
└─────────────────────────────────────┘
```

---

## Adding a New Model: Before vs After

### BEFORE (Old train.py)

To add a new model called "AutoEncoder":

1. Write a new training function (30-50 lines):
```python
def train_autoencoder(X_train, hyperparams):
    model = AutoEncoder(
        layers=hyperparams.get('layers', [64, 32, 64]),
        activation=hyperparams.get('activation', 'relu'),
        # ... 20 more lines
    )
    model.fit(X_train)
    return model
```

2. Update the if/elif chain in main():
```python
if model_type == 'isolation_forest':
    model = train_isolation_forest(X_train, hyperparams)
elif model_type == 'knn_lof':
    model = train_knn_lof(X_train, hyperparams)
elif model_type == 'one_class_svm':
    model = train_one_class_svm(X_train, hyperparams)
elif model_type == 'autoencoder':  # ← Add this
    model = train_autoencoder(X_train, hyperparams)
```

3. Update evaluation function to handle new model's scoring method

**Total**: ~60-80 lines of changes in 3 different places

---

### AFTER (New train_refactored.py)

To add "AutoEncoder":

1. Write a tiny class (10-15 lines):
```python
class AutoEncoderTrainer(ModelTrainer):
    def _create_model(self):
        return AutoEncoder(
            layers=self.hyperparams.get('layers', [64, 32, 64]),
            activation=self.hyperparams.get('activation', 'relu'),
            learning_rate=self.hyperparams.get('learning_rate', 0.001)
        )
```

2. Add one line to the dictionary:
```python
MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer,
    'autoencoder': AutoEncoderTrainer  # ← Add this
}
```

**Total**: ~12 lines in ONE place. Everything else (load, prep, evaluate, save) is inherited!

---

## Real-World Analogy

### Functional Approach (Old)
Like having separate instruction manuals for each appliance:
- **Coffee maker**: 1. Plug in, 2. Add water, 3. Add coffee, 4. Press button, 5. Pour, 6. Clean
- **Blender**: 1. Plug in, 2. Add ingredients, 3. Add liquid, 4. Press button, 5. Pour, 6. Clean
- **Toaster**: 1. Plug in, 2. Add bread, 3. —, 4. Press button, 5. Remove, 6. Clean

Notice the repetition? Steps 1, 4, 5, 6 are nearly identical!

---

### Class-Based Approach (New)
Like having ONE master appliance manual with a "common steps" section:

**Master Manual (ModelTrainer)**:
- Step 1: Plug in (power up)
- Step 2: Prepare (SEE SPECIFIC APPLIANCE)
- Step 3: Add ingredients (SEE SPECIFIC APPLIANCE)
- Step 4: Activate (press button)
- Step 5: Retrieve result
- Step 6: Clean up

**Coffee Maker Manual**: Only says "Step 2: Add water, Step 3: Add coffee"
**Blender Manual**: Only says "Step 2: Add liquid, Step 3: Add ingredients"
**Toaster Manual**: Only says "Step 2: Skip, Step 3: Add bread"

Much less repetition! Each appliance only documents what makes it unique.

---

## Key Class Concepts

### 1. `self` - The Object Itself
When you see `self`, it means "this specific trainer object".

```python
class ModelTrainer:
    def __init__(self, config_path):
        self.config = load_yaml(config_path)  # Store config
        self.model = None  # Will be set later

    def train(self):
        self.model = self._create_model()  # Access stored data
        self.model.fit(self.X_train)       # Use stored data
```

Think of `self` as a backpack that carries all the data with it:
- `self.config` = config file in the backpack
- `self.model` = trained model in the backpack
- `self.X_train` = training data in the backpack

Every method can access the backpack!

---

### 2. `__init__()` - The Constructor
This runs **automatically** when you create an object. It's the setup phase.

```python
# When you do this:
trainer = IsolationForestTrainer("config.yaml")

# Python automatically runs:
IsolationForestTrainer.__init__(self, "config.yaml")
```

It's like filling the backpack before you start hiking.

---

### 3. Methods - Functions Inside a Class
A **method** is just a function that belongs to a class.

```python
class ModelTrainer:
    def load_data(self):  # ← This is a method
        return pd.read_parquet(...)

    def train(self):  # ← This is also a method
        data = self.load_data()  # Call another method
```

Methods can call each other using `self.method_name()`.

---

### 4. `_create_model()` - The Template Method Pattern
Notice the underscore `_` prefix? This signals "internal method, not for external use".

```python
class ModelTrainer:
    def _create_model(self):
        raise NotImplementedError("Must override")

class IsolationForestTrainer(ModelTrainer):
    def _create_model(self):  # Override the blank template
        return IsolationForest(...)
```

This pattern is called **Template Method**: the parent defines the algorithm structure (`train()`), but leaves specific steps blank for children to fill in (`_create_model()`).

---

## The Factory Pattern: `MODEL_TRAINERS` Dictionary

```python
MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer
}

# Usage:
model_type = 'isolation_forest'
trainer_class = MODEL_TRAINERS[model_type]  # Get the class
trainer = trainer_class(config_path)         # Create an object
```

This is called the **Factory Pattern**: a dictionary that "manufactures" the right object based on input.

**Analogy**: A vending machine:
- Press button A1 → get chips
- Press button B2 → get soda
- Press button C3 → get candy

```python
VENDING_MACHINE = {
    'A1': Chips,
    'B2': Soda,
    'C3': Candy
}

selection = 'A1'
snack = VENDING_MACHINE[selection]()  # Get chips
```

---

## Why This is Better

### 1. **Less Duplication**
Common code (load, prepare, evaluate, save) is written **once** in the base class.

### 2. **Easier to Extend**
Adding a new model is 10 lines instead of 60.

### 3. **Easier to Test**
You can test each class independently:
```python
def test_isolation_forest_trainer():
    trainer = IsolationForestTrainer(test_config)
    trainer.train()
    assert trainer.model is not None
```

### 4. **Reusable**
You can use trainers in other scripts:
```python
# In a Jupyter notebook:
from train_refactored import IsolationForestTrainer

trainer = IsolationForestTrainer('config.yaml')
trainer.train()

# Access the trained model:
predictions = trainer.model.predict(new_data)
```

### 5. **Clearer Organization**
Related functionality is grouped together in a class, not scattered across functions.

---

## Line Count Comparison

| File | Lines | Comments |
|------|-------|----------|
| `train.py` | 360 | Original functional approach |
| `train_refactored.py` | ~310 | Class-based approach |

**Wait, why is it similar?**
- We added LOTS of documentation comments for learning
- The logic is identical, just reorganized
- The real savings come when adding **new models** (10 lines vs 60 lines each)

---

## When to Use Classes vs Functions

### Use Functions When:
- Simple, one-off scripts
- No shared state needed
- Each function is independent

### Use Classes When:
- Many operations share common data
- You need to extend/reuse functionality
- Related operations should be grouped
- You want to inherit behavior

For `train.py`, classes are better because:
- All models share 90% of the logic
- We need to add more models in the future
- Training has multiple steps that share data

---

## Testing the Refactored Version

The refactored version works **identically** to the original:

```bash
# Old way (still works):
python scripts/train.py --config configs/models/isolation_forest.yaml

# New way (same result):
python scripts/train_refactored.py --config configs/models/isolation_forest.yaml
```

Both produce the same outputs:
- `artifacts/models/ims_iforest.joblib`
- `artifacts/models/ims_iforest_scaler.joblib`
- `artifacts/reports/ims_iforest/run.json`
- `artifacts/reports/ims_iforest/features.txt`

---

## Next Steps

1. **Test it**: Run both versions side-by-side and compare outputs
2. **Understand the flow**: Read through `ModelTrainer.train()` to see the full process
3. **Try extending it**: Add a simple new model class
4. **Decide**: Keep the refactored version if you plan to add more models

The refactored version is more "professional" and scalable, but the original is simpler to understand at first. Choose based on your needs!
