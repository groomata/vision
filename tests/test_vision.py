# import torch
# import torch.nn.functional as F
# from PIL import Image

# from groovis import Vision

# def test_invariance():

#     vision = torch.load("build/vision.pth")

#     image_tiger_1 = image_path_to_tensor("data/test/tiger_1.webp")
#     image_tiger_2 = image_path_to_tensor("data/test/tiger_2.webp")
#     image_dog = image_path_to_tensor("data/test/dog.webp")

#     tiger_1 = vision(image_tiger_1)
#     tiger_2 = vision(image_tiger_2)
#     dog = vision(image_dog)

#     diff_tiger_tiger = F.l1_loss(tiger_2, tiger_1)
#     diff_tiger_dog_1 = F.l1_loss(tiger_1, dog)
#     diff_tiger_dog_2 = F.l1_loss(tiger_2, dog)

#     quality = (diff_tiger_dog_1 + diff_tiger_dog_2) / 2 - diff_tiger_tiger

#     print(quality)

#     assert quality > 0
