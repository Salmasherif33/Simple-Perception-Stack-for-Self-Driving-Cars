import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def train(car_features , notcar_features):

    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fchange X to be a column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    # splitting the data
    rand_state = np.random.randint(0,100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, 
                                                        random_state=rand_state)
    

    #print('x_scaler',len(X_scaler[0]))
    
    # train linear SVC
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    acc = round(svc.score(X_test, y_test), 4)

    return svc , X_scaler