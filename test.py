import pickle
with open('models/emotion_model_multilabel.pkl', 'rb') as f:
    model = pickle.load(f)

print(type(model))
print(model.named_steps)
print(model.classes_)
