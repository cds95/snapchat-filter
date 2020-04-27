import cv2
import sunglasses_filter, rainbow_filter, dog_filter

vc = cv2.VideoCapture(0)
rainbow_filterer = rainbow_filter.RainbowFilter()
sunglasses_filterer = sunglasses_filter.SunglassesFilter()
dog_filterer = dog_filter.DogFilter()

while(True):
    ret, frame = vc.read()
    try:
        filtered_img = dog_filterer.apply(frame)
        filtered_img = rainbow_filterer.apply(filtered_img)
        filtered_img = sunglasses_filterer.apply(filtered_img)
        cv2.imshow('frame', filtered_img)
    except:
        print("Failed to filter")
        cv2.imshow('frame', frame)

    # This breaks on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
