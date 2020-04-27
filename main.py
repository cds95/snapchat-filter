import cv2
import sunglasses_filter, rainbow_filter, dog_filter

vc = cv2.VideoCapture(0)
rainbow_filterer = rainbow_filter.RainbowFilter()
sunglasses_filterer = sunglasses_filter.SunglassesFilter()
dog_filterer = dog_filter.DogFilter()

filters = [dog_filterer, sunglasses_filterer, rainbow_filterer] # Add filters to list to apply them

while(True):
    ret, frame = vc.read()
    try:
        filtered_img = frame
        for sp_filter in filters:
            filtered_img = sp_filter.apply(filtered_img)
        cv2.imshow('frame', filtered_img)
    except:
        print("Failed to filter")
        cv2.imshow('frame', frame)

    # This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
