import cv2

# Maskeyi yükle
mask = cv2.imread('utils/fmgproducts/landscape/mask.png')

# Gaussian blur uygulayarak kenarları yumuşat
blurred_mask = cv2.GaussianBlur(mask, (31, 31), sigmaX=0)

# Sonucu kaydet
cv2.imwrite('utils/fmgproducts/landscape/mask_blurred_2.png', blurred_mask)
