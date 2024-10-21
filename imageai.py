from imageai.Detection.Custom import DetectionModelTrainer


trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory='imageai_data')

# trainer.evaluateModel(model_path="imageai_data/models", json_path="imageai_data/json/detection_config.json",
#                      iou_threshold=0.1, object_threshold=0.1, nms_threshold=0.1)


trainer.setTrainConfig(object_names_array=["crack"], batch_size=2, num_experiments=1)

trainer.trainModel()
