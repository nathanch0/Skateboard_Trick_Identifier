from PIL import Image


"""
This helper function will re-size and preprocess a list of photos to be used
in the training set of the data
"""

def image_resize(image_list, mode='resize'):
    """
    This will resize an image in the list, and save it in a new location

    Args:
        image_list == List of image paths
        mode == Do you want to process it differently?
                resize: Resizes the image to 299x299
                distort: Makes images black and white
                rotate: Rotates images 45 degrees
    Return:
        None
    """
    if type(image_list) == 'dict':
        if mode == 'resize':
            for key, value in image_list.items():
                for element in value:
                    new_image = Image.open(element)
                    new_image.thumbnail((299,299))
                    if key == 'ollie':
                        new_image.save('train_photos/ollie/new_' + element.split('/')[-1])
                    elif key == 'kickflip':
                        new_image.save('train_photos/kickflip/new_' + element.split('/')[-1])
                print('All photos have been re-sized')

        elif mode == 'distort':
            for key, value in image_list.items():
                for element in value:
                    image = Image.open(element)
                    #new_image = image.thumbnail((299,299))
                    distorted = image.convert('L')
                    if key == 'ollie':
                        distorted.save('train_photos/ollie/color_' + element.split('/')[-1])
                    elif key == 'kickflip':
                        distorted.save('train_photos/ollie/color_' + element.split('/')[-1])
                print('All photos have been re-colored')

        elif mode == 'rotate':
            for key, value in image_list.items():
                for element in value:
                    image = Image.open(element)
                    #new_image = image.thumbnail((299,299))
                    rotated = image.rotate(45)
                    if key == 'ollie':
                        rotated.save('train_photos/ollie/rotate_' + element.split('/')[-1])
                    elif key == 'kickflip':
                        rotated.save('train_photos/kickflip/rotate_' + element.split('/')[-1])
                print('All photos have been rotated')

    elif len(image_list) == 1:
        new_image = Image.open(image_list[0])
        new_image.thumbnail((299,299))
        new_image.save('../one_prediction/predicted.jpg')
        return new_image
