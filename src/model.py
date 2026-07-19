import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def build_ann(input_shape, config):
    """
    Build and compile a TensorFlow/Keras artificial neural network
    with BatchNormalization and Dropout regularizations.
    """
    ann_cfg = config['models']['ann']
    hidden_units = ann_cfg['hidden_units']
    dropout_rates = ann_cfg['dropout_rates']
    learning_rate = ann_cfg['learning_rate']
    
    inp = Input(shape=(input_shape,))
    x = BatchNormalization()(inp)
    x = Dropout(dropout_rates[0])(x)
    
    for i in range(len(hidden_units)):
        x = Dense(hidden_units[i], activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rates[i + 1])(x)
        
    out = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[AUC(name='AUC')]
    )
    return model

def build_xgboost(config):
    """Build and instantiate XGBoost Classifier."""
    xgb_cfg = config['models']['xgboost']
    return XGBClassifier(
        use_label_encoder=xgb_cfg['use_label_encoder'],
        eval_metric=xgb_cfg['eval_metric'],
        random_state=xgb_cfg['random_state']
    )

def build_random_forest(config):
    """Build and instantiate Random Forest Classifier."""
    rf_cfg = config['models']['random_forest']
    return RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'],
        random_state=rf_cfg['random_state'],
        n_jobs=-1
    )
