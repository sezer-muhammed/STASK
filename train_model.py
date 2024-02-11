from ultralytics import YOLO, checks, hub
checks()

hub.login('82586760647651968d7037213bafd86a120860117b')

model = YOLO('https://hub.ultralytics.com/models/PkgXuLAHqZmTy8vLC7Gv')
results = model.train()