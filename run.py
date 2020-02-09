from training.train_detectron2 import train_detectron2
from evaluation.eval_detectron2 import get_detectron2_predictor


config = train_detectron2()
predictor = get_detectron2_predictor(config)


# ds = dataset_storage['synthetic'][MODES[1]]
# ds = dataset_storage['real'][MODES[1]]
# meta = MetadataCatalog.get(SYNTHETIC_DATASET_NAME[MODES[1]])
# for d in random.sample(ds['dicts'], 3):
#     im = cv.imread(d['file_name'])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=meta,
#                    scale=0.8,
#                    # instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
#     )
#     v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
#     cv2_imshow(v.get_image()[:, :, ::-1])
