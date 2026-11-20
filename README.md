# Develop a Convolutional Deep Neural Network for Image Classification
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))
~~~
~~~
t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
~~~
~~~
with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")
~~~


### OUTPUT

## Training Loss per Epoch
<img width="719" height="367" alt="image" src="https://github.com/user-attachments/assets/132baecd-d519-4d65-bbab-2c55e30eaade" />


## Confusion Matrix

<img width="819" height="590" alt="image" src="https://github.com/user-attachments/assets/49838aeb-5743-4c17-b2a1-54e416c9ca6e" />


## Classification Report

<img width="725" height="350" alt="image" src="https://github.com/user-attachments/assets/20db8557-b176-4aa0-87eb-0303f926890f" />

### New Sample Data Prediction
<img width="570" height="555" alt="image" src="https://github.com/user-attachments/assets/8f9242d4-0b61-48c4-ae57-7299386f1c0f" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.

