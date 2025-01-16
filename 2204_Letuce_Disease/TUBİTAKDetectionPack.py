import os
try:
    import cv2
    import numpy as np
    import random
    from deap import base, creator, tools, algorithms
    import tkinter as tk
    from tkinter import filedialog
except:
    os.system('pip install opencv-python')
    os.system('pip install nunpy')
    os.system('pip install deap')
    os.system('pip install random')
    os.system('pip install tk-tools')




creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()




class DecicionDetection:
    def __init__(self, healtyPlant, DiseasedPlant):
        self.hPname = healtyPlant
        self.dPname = DiseasedPlant
        
        self.image = cv2.imread(healtyPlant)  # Orijinal Marulun bulunduğu görüntüsü
        self.disease_Ltc = cv2.imread(DiseasedPlant) # Strese Maruz Kalmış Marulun görüntüsü
        self.ground_truth = cv2.imread('ground_truth.jpg', 0) # Doğrulama Maskesi
    
    def calculate_fitness(self, individual, image, ground_truth):
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

    def create_individual(self):
        h_min = random.randint(20, 70)
        h_max = random.randint(h_min + 10, 90)
        s_min = random.randint(50, 100)
        s_max = random.randint(s_min + 50, 255)
        v_min = random.randint(50, 100)
        v_max = random.randint(v_min + 50, 255)
        return [h_min, s_min, v_min, h_max, s_max, v_max]
    
    def run_genetic_algorithm(self):
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

    def takeeGrounthTruth(self):
        image = cv2.imread(self.hPname)

        # ROI (Region of Interest) seçimi
        print("Lütfen marul bölgesini seçmek için bir dikdörtgen çizin.")
        roi = cv2.selectROI("Görüntü", image, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        # ROI koordinatları
        x, y, w, h = map(int, roi)

        # Maske oluşturma
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Siyah bir maske
        mask[y:y+h, x:x+w] = 255  # Seçilen alanı beyaz yap

        # Maskeyi göster
        cv2.imshow('Mask', mask)
        cv2.imwrite('ground_truth.jpg', mask)  # Maskeyi kaydet
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def foundBestİnd(self):

        İndivisualTable = dict()
        for i in range(0,12):
            best_individual = DecicionDetection(self.hPname,self.dPname).run_genetic_algorithm()
            
            optimal_bounds =  best_individual

            # En iyi maskeyi oluştur
            lower_bound = np.array(optimal_bounds[:3], dtype=np.float64)
            upper_bound = np.array(optimal_bounds[3:], dtype=np.float64)

            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
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
        hsv_image = cv2.cvtColor(self.disease_Ltc, cv2.COLOR_BGR2HSV)
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
