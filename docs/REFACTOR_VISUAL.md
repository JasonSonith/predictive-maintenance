# Visual Guide: How the Refactored Code Works

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelTrainer (Base Class)                │
│                  "The Master Cookie Cutter"                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  __init__(config_path)                                      │
│    ↳ Load configs, set up variables                        │
│                                                             │
│  load_data()                                                │
│    ↳ Load feature parquet file                             │
│                                                             │
│  filter_normal_baseline(df)                                 │
│    ↳ Keep only "healthy" data for training                 │
│                                                             │
│  select_features(df)                                        │
│    ↳ Pick which columns to use                             │
│                                                             │
│  prepare_data()                                             │
│    ↳ Orchestrate: load → filter → select → scale           │
│                                                             │
│  _create_model()  ⚠️ BLANK - children fill this in         │
│                                                             │
│  train()  ⭐ MAIN METHOD                                     │
│    ↳ prepare → create model → fit → evaluate → save        │
│                                                             │
│  evaluate()                                                 │
│    ↳ Test model on training data                           │
│                                                             │
│  save(stats)                                                │
│    ↳ Save model, scaler, metadata                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │ Inherits from
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         │                 │                 │
┌────────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐
│ Isolation       │ │   kNN-LOF   │ │  One-Class     │
│ Forest          │ │   Trainer   │ │  SVM Trainer   │
│ Trainer         │ │             │ │                │
├─────────────────┤ ├─────────────┤ ├────────────────┤
│ _create_model() │ │_create_model│ │_create_model() │
│   ↳ return      │ │  ↳ return   │ │  ↳ return      │
│   Isolation     │ │  LOF(...)   │ │  SVM(...)      │
│   Forest(...)   │ │             │ │                │
└─────────────────┘ └─────────────┘ └────────────────┘
   10 lines           10 lines         10 lines
```

---

## Data Flow When You Run the Code

```
USER RUNS:
$ python train_refactored.py --config configs/models/isolation_forest.yaml

    ↓

MAIN FUNCTION:
1. Reads config file to find model_type = "isolation_forest"
2. Looks up in dictionary: MODEL_TRAINERS['isolation_forest'] → IsolationForestTrainer
3. Creates trainer object: trainer = IsolationForestTrainer(config)
4. Calls: trainer.train()

    ↓

TRAIN() METHOD (from ModelTrainer base class):
┌──────────────────────────────────────┐
│ trainer.train()                      │
├──────────────────────────────────────┤
│                                      │
│ Step 1: X_train = prepare_data()    │────→ Loads, filters, scales data
│            ↓                         │
│         [10,000 x 87]                │
│                                      │
│ Step 2: model = _create_model()     │────→ Calls CHILD's method
│            ↓                         │      (IsolationForestTrainer)
│      IsolationForest()               │
│                                      │
│ Step 3: model.fit(X_train)          │────→ Trains the model
│            ↓                         │
│      [Model trained!]                │
│                                      │
│ Step 4: eval_stats = evaluate()     │────→ Tests on training data
│            ↓                         │
│      {n_normal: 9000,                │
│       n_anomaly: 1000}               │
│                                      │
│ Step 5: save(training_stats)        │────→ Saves everything
│            ↓                         │
│   - model.joblib                     │
│   - scaler.joblib                    │
│   - run.json                         │
│   - features.txt                     │
│                                      │
└──────────────────────────────────────┘
```

---

## The Dictionary Selector (Factory Pattern)

```python
MODEL_TRAINERS = {
    'isolation_forest': IsolationForestTrainer,
    'knn_lof': KNNLOFTrainer,
    'one_class_svm': OneClassSVMTrainer
}
```

Think of this as a vending machine:

```
┌─────────────────────────────────────────┐
│        MODEL TRAINERS VENDING MACHINE   │
├─────────────────────────────────────────┤
│                                         │
│  [A1] isolation_forest                  │
│       → IsolationForestTrainer          │
│                                         │
│  [B2] knn_lof                           │
│       → KNNLOFTrainer                   │
│                                         │
│  [C3] one_class_svm                     │
│       → OneClassSVMTrainer              │
│                                         │
└─────────────────────────────────────────┘

USER INPUT: "isolation_forest"
    ↓
MACHINE: "Here's your IsolationForestTrainer!"
    ↓
CREATE OBJECT: trainer = IsolationForestTrainer(config)
```

---

## How Inheritance Works: The Backpack Analogy

```
┌───────────────────────────────────────────────────────────┐
│                      ModelTrainer                         │
│                  (Parent/Base Class)                      │
│                                                           │
│  🎒 BACKPACK (self):                                      │
│     - self.config                                         │
│     - self.dataset_config                                 │
│     - self.hyperparams                                    │
│     - self.model                                          │
│     - self.scaler                                         │
│     - self.X_train                                        │
│     - self.feature_cols                                   │
│                                                           │
│  🛠️ TOOLS (methods):                                      │
│     - load_data()                                         │
│     - prepare_data()                                      │
│     - train()                                             │
│     - evaluate()                                          │
│     - save()                                              │
│                                                           │
└───────────────────────────────────────────────────────────┘
                         ▲
                         │
                    Inherits
                         │
┌────────────────────────┴──────────────────────────────────┐
│            IsolationForestTrainer                         │
│                (Child Class)                              │
│                                                           │
│  Gets EVERYTHING from parent automatically:               │
│    ✅ The backpack (all data)                             │
│    ✅ All the tools (all methods)                         │
│                                                           │
│  Only needs to add what's unique:                         │
│    🆕 _create_model() → IsolationForest(...)              │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

When you call `trainer.train()`:
1. Python looks for `train()` in IsolationForestTrainer → Not found
2. Python looks in parent ModelTrainer → Found! Use that
3. Inside `train()`, it calls `self._create_model()`
4. Python looks for `_create_model()` in IsolationForestTrainer → Found! Use that
5. Returns IsolationForest model

---

## Comparing Old vs New: Adding a Model

### OLD APPROACH (train.py)

```
To add "AutoEncoder":

📝 Step 1: Write training function (40 lines)
   ┌─────────────────────────────────────┐
   │ def train_autoencoder(X, params):   │
   │     # 40 lines of code              │
   │     model = AutoEncoder(...)        │
   │     model.fit(X)                    │
   │     return model                    │
   └─────────────────────────────────────┘

📝 Step 2: Update main() if/elif chain (5 lines)
   ┌─────────────────────────────────────┐
   │ if model_type == 'isolation_forest':│
   │     model = train_isolation_forest()│
   │ elif model_type == 'knn_lof':       │
   │     model = train_knn_lof()         │
   │ elif model_type == 'autoencoder':   │ ← ADD THIS
   │     model = train_autoencoder()     │ ← ADD THIS
   └─────────────────────────────────────┘

📝 Step 3: Update evaluate() (10 lines)
   ┌─────────────────────────────────────┐
   │ if model_type == 'autoencoder':     │ ← ADD THIS
   │     scores = model.get_scores()     │ ← ADD THIS
   └─────────────────────────────────────┘

TOTAL: ~55 lines across 3 different locations
```

### NEW APPROACH (train_refactored.py)

```
To add "AutoEncoder":

📝 Step 1: Write tiny class (12 lines)
   ┌─────────────────────────────────────┐
   │ class AutoEncoderTrainer(           │
   │     ModelTrainer):                  │
   │                                     │
   │     def _create_model(self):        │
   │         return AutoEncoder(         │
   │             layers=self.hyperparams │
   │                 .get('layers'),     │
   │             learning_rate=self      │
   │                 .hyperparams        │
   │                 .get('lr')          │
   │         )                           │
   └─────────────────────────────────────┘

📝 Step 2: Add to dictionary (1 line)
   ┌─────────────────────────────────────┐
   │ MODEL_TRAINERS = {                  │
   │     'isolation_forest': ...,        │
   │     'knn_lof': ...,                 │
   │     'autoencoder':                  │ ← ADD THIS
   │         AutoEncoderTrainer          │ ← ADD THIS
   │ }                                   │
   └─────────────────────────────────────┘

TOTAL: ~12 lines in ONE location
Everything else inherited! ✨
```

---

## The Template Method Pattern

```
ModelTrainer.train() defines the ALGORITHM STRUCTURE:

┌─────────────────────────────────────────────────┐
│ def train(self):                                │
│                                                 │
│   1. prepare_data()      ← Common (from parent)│
│                                                 │
│   2. _create_model()     ← Variable (from child)│
│                                                 │
│   3. model.fit()         ← Common (from parent)│
│                                                 │
│   4. evaluate()          ← Common (from parent)│
│                                                 │
│   5. save()              ← Common (from parent)│
│                                                 │
└─────────────────────────────────────────────────┘

Only step 2 changes per model!
The recipe is fixed, one ingredient varies.
```

**Analogy**: Making different sandwiches

1. Get bread → Same for all
2. **Add filling** → Different (turkey, veggie, PB&J)
3. Add condiments → Same for all
4. Cut in half → Same for all
5. Wrap and serve → Same for all

Only step 2 varies!

---

## Memory (Self) Visualization

```
When you create a trainer:

trainer = IsolationForestTrainer("config.yaml")

Python creates an object in memory:

┌──────────────────────────────────────────┐
│  IsolationForestTrainer @ 0x7f8a3b4c     │
├──────────────────────────────────────────┤
│  ATTRIBUTES (self.___):                  │
│                                          │
│  self.config_path = "config.yaml"        │
│  self.model_config = {                   │
│      'model_type': 'isolation_forest',   │
│      'hyperparameters': {...}            │
│  }                                       │
│  self.dataset_config = {...}             │
│  self.model_type = 'isolation_forest'    │
│  self.hyperparams = {...}                │
│  self.model = None  (will be set later)  │
│  self.scaler = None (will be set later)  │
│  self.feature_cols = None                │
│  self.X_train = None                     │
│                                          │
│  METHODS (inherited from ModelTrainer):  │
│    - load_data()                         │
│    - prepare_data()                      │
│    - train()                             │
│    - evaluate()                          │
│    - save()                              │
│    - _create_model() [overridden]        │
│                                          │
└──────────────────────────────────────────┘

All methods can access all attributes via "self"!

Example:
  def train(self):
      X = self.prepare_data()     ← Access stored method
      self.model = self._create_model()  ← Access attribute
      self.model.fit(X)           ← Use stored data
```

---

## Class vs Function: Data Passing

### FUNCTIONS (Old way):
```
┌─────────────────────────────────────────┐
│ config = load_config(path)              │
│    ↓                                    │
│ df = load_data(config)                  │
│    ↓                                    │
│ df_train = filter(df, config)           │
│    ↓                                    │
│ X = select_features(df_train, config)   │
│    ↓                                    │
│ model = train_if(X, config['params'])   │
│    ↓                                    │
│ stats = evaluate(model, X)              │
│    ↓                                    │
│ save(model, config, stats)              │
└─────────────────────────────────────────┘

Notice: Must pass data to EVERY function!
```

### CLASS (New way):
```
┌─────────────────────────────────────────┐
│ trainer = Trainer(path)                 │
│    ↓ (stores config in self)            │
│                                         │
│ trainer.train()                         │
│   ├─ self.prepare_data()                │
│   │    (uses self.config)               │
│   ├─ self._create_model()               │
│   │    (uses self.hyperparams)          │
│   ├─ self.model.fit(self.X_train)       │
│   ├─ self.evaluate()                    │
│   │    (uses self.model, self.X_train)  │
│   └─ self.save()                        │
│        (uses self.model, self.config)   │
└─────────────────────────────────────────┘

Everything is stored in "self" - no passing needed!
```

---

## When to Use Each Approach

### Use FUNCTIONS (Old) when:
- ✅ Simple, one-time script
- ✅ Few functions (< 5)
- ✅ No shared state
- ✅ Linear flow
- ✅ Learning/prototyping

### Use CLASSES (New) when:
- ✅ Complex workflow (> 5 steps)
- ✅ Lots of shared data
- ✅ Multiple variants (models, datasets)
- ✅ Need to extend/reuse
- ✅ Production code

**For train.py**: Classes are better because:
- 3 models (growing to 4+)
- 7+ steps with shared data
- Need to add more models
- Production pipeline

---

## Summary: The Key Insight

```
OLD:
  write_cookies_recipe_for_chocolate_chip()  ← 50 lines
  write_cookies_recipe_for_oatmeal()        ← 50 lines
  write_cookies_recipe_for_sugar()          ← 50 lines
  TOTAL: 150 lines, lots of duplication

NEW:
  class CookieRecipe:                       ← 40 lines (common steps)
      mix, bake, cool, package

  class ChocolateChip(CookieRecipe):        ← 5 lines (unique part)
      add chocolate chips

  class Oatmeal(CookieRecipe):              ← 5 lines (unique part)
      add oats

  class Sugar(CookieRecipe):                ← 5 lines (unique part)
      add sugar

  TOTAL: 55 lines, NO duplication
```

That's the power of classes! 🎉
