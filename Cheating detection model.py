

import cv2
import numpy as np
import time
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.optimizers import Adam

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────
# 1. LFW DATASET
# ─────────────────────────────────────────────────────────────
print("Fetching LFW Dataset...")
lfw_dataset  = fetch_lfw_people(min_faces_per_person=70, resize=0.5)
lfw_faces    = [np.uint8(img * 255) for img in lfw_dataset.images]
lfw_labels   = list(lfw_dataset.target_names[i] for i in lfw_dataset.target)
target_names = list(lfw_dataset.target_names)

AUTHORIZED_LABEL = "Authorized_Student"
target_names.append(AUTHORIZED_LABEL)
print(f"LFW Loaded! {len(lfw_faces)} background faces.")

# ─────────────────────────────────────────────────────────────
# 2. OPENCV DETECTORS + YOLO
# ─────────────────────────────────────────────────────────────
face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade     = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

print("Loading YOLOv8n for object detection...")
yolo_model = YOLO('yolov8n.pt')
FORBIDDEN_OBJECTS = ['cell phone', 'book', 'laptop']

# ─────────────────────────────────────────────────────────────
# 3. CONSTANTS
# ─────────────────────────────────────────────────────────────
IMAGE_SIZE       = 96   # MobileNetV2 works well at 96x96 (min 32)
BATCH_SIZE       = 32
EPOCHS_HEAD      = 10   # Train only the new head (base frozen)
EPOCHS_FINETUNE  = 5    # Fine-tune top layers of MobileNetV2

# ─────────────────────────────────────────────────────────────
# 4. BUILD TRANSFER LEARNING CNN (MobileNetV2 backbone)
# ─────────────────────────────────────────────────────────────
def build_transfer_cnn(num_classes, image_size):
    """
    Transfer Learning CNN:
      Base  : MobileNetV2 pretrained on ImageNet (weights frozen initially)
      Head  : GlobalAvgPool → Dense(256) → Dropout → Dense(128) → Softmax

    Training strategy:
      Step 1 — Freeze base, train head only   (fast, stable)
      Step 2 — Unfreeze top 30 layers, fine-tune at low LR (improves accuracy)
    """
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,                          # Remove ImageNet classifier
        input_shape=(image_size, image_size, 3)
    )
    base_model.trainable = False                    # Freeze all base layers initially

    # Build custom head on top
    inputs = Input(shape=(image_size, image_size, 3))
    x = base_model(inputs, training=False)          # training=False keeps BN frozen
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu',
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model, base_model


def unfreeze_top_layers(model, base_model, num_layers=30, lr=1e-5):
    """
    Unfreeze the top `num_layers` of the MobileNetV2 base for fine-tuning.
    Uses a very low learning rate to avoid destroying pretrained features.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(f"Fine-tuning: {num_layers} top layers unfrozen, LR={lr}")
    return model

# ─────────────────────────────────────────────────────────────
# 5. TRAIN + EVALUATE TRANSFER CNN
# ─────────────────────────────────────────────────────────────
def train_and_evaluate_transfer_cnn(lfw_faces, lfw_labels, student_frames, authorized_label):
    save_dir = "temp_face_data_v2"
    os.makedirs(save_dir, exist_ok=True)

    image_paths, labels = [], []

    print("Saving LFW faces...")
    for idx, (face_img, label) in enumerate(zip(lfw_faces, lfw_labels)):
        resized = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        path    = os.path.join(save_dir, f"lfw_{idx}.jpg")
        cv2.imwrite(path, rgb)
        image_paths.append(path); labels.append(label)

    print("Saving student frames...")
    for idx, face_img in enumerate(student_frames):
        resized = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        path    = os.path.join(save_dir, f"student_{idx}.jpg")
        cv2.imwrite(path, rgb)
        image_paths.append(path); labels.append(authorized_label)

    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    # MobileNetV2 needs its own preprocessing ([-1, 1] range)
    datagen_train = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.2,
        horizontal_flip=True, fill_mode='nearest'
    )
    datagen_val = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)

    train_gen = datagen_train.flow_from_dataframe(
        dataframe=train_df, x_col='image_path', y_col='label',
        target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
        shuffle=True, class_mode='categorical'
    )
    val_gen = datagen_val.flow_from_dataframe(
        dataframe=val_df, x_col='image_path', y_col='label',
        target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE,
        shuffle=False, class_mode='categorical'
    )

    class_indices = train_gen.class_indices
    index_to_name = {v: k for k, v in class_indices.items()}
    class_names   = [index_to_name[i] for i in range(len(index_to_name))]

    model, base_model = build_transfer_cnn(len(class_indices), IMAGE_SIZE)
    model.summary()

    # ── STEP 1: Train head only ───────────────────────────────
    print("\n[Step 1] Training head (base frozen)...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history1   = model.fit(train_gen, validation_data=val_gen,
                           epochs=EPOCHS_HEAD, callbacks=[early_stop])

    # ── STEP 2: Fine-tune top layers ──────────────────────────
    print("\n[Step 2] Fine-tuning top MobileNetV2 layers...")
    model = unfreeze_top_layers(model, base_model, num_layers=30, lr=1e-5)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    history2   = model.fit(train_gen, validation_data=val_gen,
                           epochs=EPOCHS_FINETUNE,
                           callbacks=[early_stop, reduce_lr])

    # Combine histories
    combined_acc     = history1.history['accuracy']     + history2.history['accuracy']
    combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    combined_loss    = history1.history['loss']         + history2.history['loss']
    combined_val_loss= history1.history['val_loss']     + history2.history['val_loss']

    # ── CONFUSION MATRIX ──────────────────────────────────────
    print("\nGenerating Confusion Matrix...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)
    y_true       = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names)-2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('VERSION 2 — Transfer CNN (MobileNetV2): Face Recognition Confusion Matrix',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix_v2_face_cnn.png", dpi=150)
    plt.close()
    print("Confusion matrix saved → confusion_matrix_v2_face_cnn.png")

    # ── CLASSIFICATION REPORT ─────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print("\nCLASSIFICATION REPORT (Transfer CNN — V2):\n")
    print(report)

    # ── TRAINING CURVES ───────────────────────────────────────
    epochs_total = list(range(1, len(combined_acc) + 1))
    head_end     = len(history1.history['accuracy'])

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_total, combined_acc,     label='Train Acc')
    axes[0].plot(epochs_total, combined_val_acc, label='Val Acc')
    axes[0].axvline(x=head_end, color='gray', linestyle='--', label='Fine-tune start')
    axes[0].set_title('Accuracy'); axes[0].legend()

    axes[1].plot(epochs_total, combined_loss,     label='Train Loss')
    axes[1].plot(epochs_total, combined_val_loss, label='Val Loss')
    axes[1].axvline(x=head_end, color='gray', linestyle='--', label='Fine-tune start')
    axes[1].set_title('Loss'); axes[1].legend()

    plt.suptitle('VERSION 2 — Transfer CNN (MobileNetV2) Training Curves')
    plt.tight_layout()
    plt.savefig("training_curve_v2.png", dpi=150)
    plt.close()

    with open("evaluation_v2.txt", "w") as f:
        f.write("VERSION 2 — Transfer CNN (MobileNetV2) Face Recognition\n")
        f.write("=" * 60 + "\n")
        f.write(report)

    return model, index_to_name


# ─────────────────────────────────────────────────────────────
# 6. YOLO OBJECT DETECTION EVALUATOR  (same as V1)
# ─────────────────────────────────────────────────────────────
class YOLOEvaluator:
    def __init__(self, class_names):
        self.classes             = class_names
        self.tp                  = {c: 0 for c in class_names}
        self.fp                  = {c: 0 for c in class_names}
        self.fn                  = {c: 0 for c in class_names}
        self.total_frames        = 0
        self.false_alarm_frames  = 0

    def record(self, detected, ground_truth):
        self.total_frames += 1
        det_set = set(detected); gt_set = set(ground_truth)
        has_fp  = False
        for c in self.classes:
            if   c in det_set and c in gt_set:     self.tp[c] += 1
            elif c in det_set and c not in gt_set: self.fp[c] += 1; has_fp = True
            elif c not in det_set and c in gt_set: self.fn[c] += 1
        if has_fp and not (det_set & gt_set):
            self.false_alarm_frames += 1

    def print_report(self):
        lines = ["\n" + "="*60,
                 "YOLO OBJECT DETECTION EVALUATION REPORT (Version 2)",
                 "="*60,
                 f"Total frames evaluated : {self.total_frames}",
                 f"False-alarm frames     : {self.false_alarm_frames}"]
        if self.total_frames:
            lines.append(f"False-alarm rate       : {self.false_alarm_frames/self.total_frames:.2%}")
        lines += ["",
                  f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}",
                  "-"*60]
        all_tp = all_fp = all_fn = 0
        for c in self.classes:
            tp,fp,fn = self.tp[c],self.fp[c],self.fn[c]
            prec = tp/(tp+fp) if (tp+fp) else 0.0
            rec  = tp/(tp+fn) if (tp+fn) else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
            lines.append(f"{c:<15} {prec:>10.2%} {rec:>10.2%} {f1:>10.2%} {tp:>6} {fp:>6} {fn:>6}")
            all_tp+=tp; all_fp+=fp; all_fn+=fn
        lines.append("-"*60)
        tp,fp,fn = all_tp,all_fp,all_fn
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        lines.append(f"{'OVERALL':<15} {prec:>10.2%} {rec:>10.2%} {f1:>10.2%} {tp:>6} {fp:>6} {fn:>6}")
        lines.append("="*60)
        rpt = "\n".join(lines)
        print(rpt)
        with open("evaluation_v2.txt","a") as f:
            f.write("\n\nYOLO Object Detection Evaluation\n" + rpt)

        # Per-class confusion matrices
        fig,axes = plt.subplots(1,len(self.classes),figsize=(4*len(self.classes),4))
        if len(self.classes)==1: axes=[axes]
        for ax,c in zip(axes,self.classes):
            tp,fp,fn = self.tp[c],self.fp[c],self.fn[c]
            tn = self.total_frames-tp-fp-fn
            sns.heatmap(np.array([[tn,fp],[fn,tp]]),annot=True,fmt='d',cmap='Oranges',ax=ax,
                        xticklabels=['Pred: No','Pred: Yes'],yticklabels=['True: No','True: Yes'])
            ax.set_title(f'YOLO — {c}')
        plt.suptitle('VERSION 2 — YOLO Object Detection Confusion Matrices')
        plt.tight_layout()
        plt.savefig("confusion_matrix_v2_yolo.png",dpi=150)
        plt.close()
        print("YOLO confusion matrices saved → confusion_matrix_v2_yolo.png")


# ─────────────────────────────────────────────────────────────
# 7. FACE DETECTION HELPER
# ─────────────────────────────────────────────────────────────
def get_all_faces(gray_frame, frame_w):
    faces     = list(face_cascade.detectMultiScale(gray_frame,1.2,5,minSize=(50,50)))
    face_type = "frontal"
    if not faces:
        profiles = list(profile_cascade.detectMultiScale(gray_frame,1.2,5,minSize=(50,50)))
        if profiles:
            faces,face_type = profiles,"profile"
        else:
            flipped = cv2.flip(gray_frame,1)
            lp = profile_cascade.detectMultiScale(flipped,1.2,5,minSize=(50,50))
            for (x,y,w,h) in lp: faces.append((frame_w-x-w,y,w,h))
            if faces: face_type="profile"
    return faces,face_type


# ─────────────────────────────────────────────────────────────
# PHASE 1: REGISTRATION
# ─────────────────────────────────────────────────────────────
print("\n--- PHASE 1: REGISTRATION ---")
cap = cv2.VideoCapture(0)
registration_frames = []
REQUIRED_FRAMES     = 90

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_h, frame_w, _ = frame.shape
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, _ = get_all_faces(gray, frame_w)
    prog = len(registration_frames)

    if   prog < 30: instruction,color = "1/3: Look STRAIGHT",      (0,255,255)
    elif prog < 60: instruction,color = "2/3: Turn slightly LEFT",  (255,100,100)
    else:           instruction,color = "3/3: Turn slightly RIGHT", (100,255,100)

    cv2.putText(frame, f"REGISTRATION: {instruction} ({prog}/{REQUIRED_FRAMES})",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if len(faces)==1:
        x,y,w,h=faces[0]
        registration_frames.append(gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
    elif len(faces)>1:
        cv2.putText(frame,"ERROR: Multiple faces!",(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.imshow('Registration V2', frame)
    if cv2.waitKey(1)&0xFF==ord('q'): print("Aborted."); break
    if len(registration_frames)>=REQUIRED_FRAMES: print("Registration complete."); break

cap.release(); cv2.destroyAllWindows()

if len(registration_frames)<REQUIRED_FRAMES:
    print("Not enough frames. Exiting."); exit()

# ─────────────────────────────────────────────────────────────
# PHASE 2: TRAIN & EVALUATE TRANSFER CNN
# ─────────────────────────────────────────────────────────────
print("\n--- PHASE 2: TRAINING TRANSFER CNN (MobileNetV2) ---")
cnn_model, index_to_name = train_and_evaluate_transfer_cnn(
    lfw_faces, lfw_labels, registration_frames, AUTHORIZED_LABEL
)
print("Transfer CNN training + evaluation complete.\n")

# ─────────────────────────────────────────────────────────────
# PHASE 3: PROCTORING + YOLO EVALUATION
# ─────────────────────────────────────────────────────────────
print("--- PHASE 3: PROCTORING (press Q to stop and see YOLO eval) ---")
cap            = cv2.VideoCapture(0)
last_eye_time  = time.time()
yolo_evaluator = YOLOEvaluator(FORBIDDEN_OBJECTS)
frame_idx      = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    frame_h, frame_w, _ = frame.shape
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    warnings = []

    # ── YOLO Detection ────────────────────────────────────────
    detected_objects = []
    for r in yolo_model(frame, verbose=False):
        for box in r.boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            class_name = yolo_model.names[cls_id]
            if class_name in FORBIDDEN_OBJECTS and conf > 0.50:
                detected_objects.append(class_name)
                warnings.append(f"FORBIDDEN: {class_name.upper()}")
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)
                cv2.putText(frame,f"{class_name} {conf:.0%}",
                            (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),2)

    simulated_gt = detected_objects if frame_idx%2==1 else []
    yolo_evaluator.record(detected_objects, simulated_gt)

    # ── Face Recognition (Transfer CNN) ───────────────────────
    faces, face_type = get_all_faces(gray, frame_w)
    eyes_this_frame  = False

    for (x,y,w,h) in faces:
        roi = gray[y:y+h,x:x+w]
        if roi.size==0: continue

        # Transfer CNN uses mobilenet_preprocess (not /255)
        face_rgb   = cv2.cvtColor(cv2.resize(roi,(IMAGE_SIZE,IMAGE_SIZE)),cv2.COLOR_GRAY2RGB)
        face_input = mobilenet_preprocess(face_rgb.astype('float32'))
        face_input = np.expand_dims(face_input, axis=0)

        preds     = cnn_model.predict(face_input, verbose=0)
        label_id  = int(np.argmax(preds,axis=1)[0])
        conf      = float(preds[0][label_id])
        pred_name = index_to_name.get(label_id,"Unknown")

        if pred_name==AUTHORIZED_LABEL and conf>0.6:
            text,color = "Authorized Student",(0,255,0)
        else:
            text,color = f"UNAUTHORIZED: {pred_name}",(0,0,255)
            warnings.append("UNAUTHORIZED PERSON")

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,f"{text} ({conf:.0%})",(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)

        if face_type=="frontal":
            ey_s=y+int(h*.20); ey_e=y+int(h*.55)
            band=gray[ey_s:ey_e,x:x+w]
            eyes=eye_cascade.detectMultiScale(band,1.1,7,
                                              minSize=(int(w*.12),int(h*.12)))
            if len(eyes):
                eyes_this_frame=True
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(frame,(x+ex,ey_s+ey),(x+ex+ew,ey_s+ey+eh),(0,255,0),1)
                    blur=cv2.GaussianBlur(band[ey:ey+eh,ex:ex+ew],(7,7),0)
                    _,th=cv2.threshold(blur,45,255,cv2.THRESH_BINARY_INV)
                    cnts,_=cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        M=cv2.moments(max(cnts,key=cv2.contourArea))
                        if M['m00']:
                            cx2=int(M['m10']/M['m00']); cy2=int(M['m01']/M['m00'])
                            cv2.circle(frame,(x+ex+cx2,ey_s+ey+cy2),3,(0,0,255),-1)
                            if cx2/ew<0.35 or cx2/ew>0.65:
                                warnings.append("EYES DIVERTED")

    if eyes_this_frame: last_eye_time=time.time()
    elif time.time()-last_eye_time>2.0: warnings.append("NO EYES DETECTED")

    if   len(faces)==0: warnings.append("NO FACE DETECTED")
    elif len(faces)>1:  warnings.append("MULTIPLE PEOPLE DETECTED")

    desk_y=int(frame_h*.60)
    hsv=cv2.cvtColor(frame[desk_y:,0:frame_w],cv2.COLOR_BGR2HSV)
    cv2.line(frame,(0,desk_y),(frame_w,desk_y),(255,0,0),1)
    mask=cv2.inRange(hsv,np.array([0,20,70]),np.array([20,255,255]))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not any(cv2.contourArea(c)>6000 for c in cnts):
        warnings.append("ARMS NOT IN DESK ZONE")

    y_off=30
    for w_txt in set(warnings):
        cv2.putText(frame,f"WARNING: {w_txt}",(20,y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        y_off+=30

    cv2.putText(frame,"VERSION 2 — MobileNetV2 Transfer CNN + YOLO",
                (10,frame_h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
    cv2.imshow('Proctoring V2 (press Q to finish + evaluate)', frame)
    if cv2.waitKey(1)&0xFF==ord('q'): break

cap.release(); cv2.destroyAllWindows()

yolo_evaluator.print_report()
print("\nAll evaluation files saved:")
print("  confusion_matrix_v2_face_cnn.png")
print("  confusion_matrix_v2_yolo.png")
print("  training_curve_v2.png")
print("  evaluation_v2.txt")
