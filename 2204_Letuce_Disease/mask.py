import cv2
import numpy as np
from deap import base, creator, tools, algorithms
import random 



# Görüntüyü yükle
image = cv2.imread('marul10.jpg')  # Marulun bulunduğu görüntü
disease_Ltc = cv2.imread('marul10.jpg')
ground_truth = cv2.imread('ground_truth.jpg', 0)  # Elle hazırlanmış doğru maske (0 = gri tonlamalı)
# Fitness fonksiyonu
def calculate_fitness(individual, image, ground_truth):
    h_min, s_min, v_min, h_max, s_max, v_max = individual

    # HSV sınır kontrolü
    if h_min > h_max or s_min > s_max or v_min > v_max:
        return 0,

    lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Eğer maske tamamen siyahsa (hiçbir piksel kapsanmıyorsa)
    if np.sum(mask) == 0:
        return 0,

    # Eğer maske tamamen beyazsa (tüm pikseller kapsanıyorsa)
    if np.sum(mask) == mask.size:
        return 0,

    # IoU hesaplama (Intersection over Union)
    intersection = np.logical_and(ground_truth, mask)
    union = np.logical_or(ground_truth, mask)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score,



# Genetik algoritma ayarları
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Birey oluşturma
def create_individual():
    h_min = random.randint(20, 70)
    h_max = random.randint(h_min + 10, 90)
    s_min = random.randint(50, 100)
    s_max = random.randint(s_min + 50, 255)
    v_min = random.randint(50, 100)
    v_max = random.randint(v_min + 50, 255)
    return [h_min, s_min, v_min, h_max, s_max, v_max]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Fitness fonksiyonunu tanımla
toolbox.register("evaluate", calculate_fitness, image=image, ground_truth=ground_truth)

# Çaprazlama, mutasyon ve seleksiyon
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=255, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_genetic_algorithm():
    population = toolbox.population(n=50)
    NGEN = 10
    CXPB = 0.7
    MUTPB = 0.1

    for gen in range(NGEN):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))

    best_ind = tools.selBest(population, k=1)[0]
    print("Best Individual: ", best_ind)
    return best_ind




# Optimizasyonu çalıştır ve sonucu uygula
def foundBestİnd():
    İndivisualTable = dict()
    for i in range(0,12):
        best_individual = run_genetic_algorithm()
        
        optimal_bounds =  best_individual

        # En iyi maskeyi oluştur
        lower_bound = np.array(optimal_bounds[:3], dtype=np.float64)
        upper_bound = np.array(optimal_bounds[3:], dtype=np.float64)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.GaussianBlur(hsv_image,(7,7),0)
        optimized_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_cleaned = cv2.morphologyEx(optimized_mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        white_pixel_count = np.count_nonzero(mask_cleaned == 255)
        black_pixel_count = np.count_nonzero(mask_cleaned == 0)
        beyaz_pixel_dagilimi = (white_pixel_count + black_pixel_count) / white_pixel_count

        İndivisualTable[beyaz_pixel_dagilimi] = best_individual

    lastİtm = sorted(İndivisualTable.keys())[9]


    optimal_bounds = İndivisualTable[lastİtm]  

    # En iyi maskeyi oluştur
    lower_bound = np.array(optimal_bounds[:3], dtype=np.float64)
    upper_bound = np.array(optimal_bounds[3:], dtype=np.float64)
    hsv_image = cv2.cvtColor(disease_Ltc, cv2.COLOR_BGR2HSV)
    hsv_image = cv2.GaussianBlur(hsv_image,(7,7),0)
    optimized_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_cleaned = cv2.morphologyEx(optimized_mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)


    whtPxlCnt = np.count_nonzero(mask_cleaned == 255)
    print('Best İndivisual: {Bestİndv}, Color Distribution: {colorDist}, White Pixel Count: {whtPxl}'.format(colorDist=lastİtm,Bestİndv=optimal_bounds,whtPxl=whtPxlCnt))
    cv2.imshow('Optimized Mask {i}'.format(i=i), mask_cleaned)


    cv2.imwrite('optimized_mask.jpg', mask_cleaned)  # Maskeyi kaydet
    cv2.waitKey(0)
    cv2.destroyAllWindows()
foundBestİnd()