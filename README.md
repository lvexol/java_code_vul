**Vulnerability Detection in Java Code Using LSTM Model**

1. **Introduction** 

Software security is a crucial concern for developers, as vulnerabilities left undetected can lead to severe exploits. Datasets such as **MegaVul** focus on C/C++ vulnerabilities, but Java, a widely used language, lacks equivalent datasets. This project converts the MegaVul dataset from C/C++ to Java using **Tree-sitter**, a parser generator tool, and trains an LSTM model to detect vulnerabilities in Java code. The goal is to create a model that automatically classifies Java code as vulnerable or non-vulnerable and build a tool that assists developers in detecting security issues early.

2. **Problem Statement** 

The MegaVul dataset, which focuses on C/C++ vulnerabilities, is not directly usable for detecting vulnerabilities in Java. Our project converts the dataset to Java using Tree-sitter, trains an LSTM model for vulnerability detection in Java, and builds a tool that classifies Java code as vulnerable or non-vulnerable. This effort provides Java developers with a reliable solution for identifying vulnerabilities. 

3. **Dataset Overview** 

MegaVul is a large-scale dataset of C/C++ vulnerabilities containing 17,380 vulnerabilities and 322,168 non-vulnerable functions (2406.12415v1). However, to apply this dataset to Java, a comprehensive conversion process was required.

4. **Dataset Conversion using Tree-sitter** 
1. ***Introduction to Tree-sitter*** 

Tree-sitter is a parser generator tool and an incremental parsing library designed to build concrete syntax trees for source files and efficiently update them. It is:

- **General**: Capable of parsing any programming language.
- **Fast**: Suitable for real-time parsing, even as source files are edited.
- **Robust**: Able to handle syntax errors gracefully. 
- **Dependency-free**: Tree-sitter is implemented in pure C, making it lightweight and easy to integrate into various applications.
2. ***Conversion Process***

We used **Tree-sitter** to convert the C/C++ MegaVul dataset to Java in the following steps:

- **Parsing C/C++ Code**: Using Tree-sitter, we generated concrete syntax trees for each C/C++ function in the MegaVul dataset. This ensured that we had a structural representation of the code.
- **Syntax Tree Analysis**: The syntax trees were analyzed to identify patterns and code structures that could be mapped to their Java equivalents.
- **Java Code Generation**: The C/C++ constructs were mapped to Java syntax using custom conversion rules. For instance:
- **Memory Management**: Vulnerabilities related to memory (buffer overflows, memory leaks) in C/C++ were mapped to Java-specific issues such as array bounds checks and NullPointerException. 
- **Error Handling**: Mapped C/C++ error handling to Java’s exception handling mechanisms. 
- **Java-Specific Augmentation**: After converting the dataset, we augmented it with Java-specific vulnerabilities, such as SQL injection, deserialization issues, and improper exception handling. 

Tree-sitter’s efficiency allowed us to quickly parse and update the syntax trees as needed, making the conversion process smoother and more accurate.

5. **Methodology** 
1. ***Data Preprocessing*** 
- **Code Cleaning**: We preprocessed the code snippets by replacing unmatched quotes and handling common issues.
- **Tokenization**: After conversion, the Java code snippets were tokenized using the javalang tokenizer, breaking them into meaningful tokens like keywords, literals, operators, etc. 
- **Padding**: Token sequences of varying lengths were padded to a uniform length for model training. 
2. ***Class Weights*** 

Due to the class imbalance (fewer vulnerable snippets), class weights were computed to ensure the model paid adequate attention to the vulnerable class during training.

6. **Model Architecture** 

The LSTM model was designed to learn patterns from tokenized Java code sequences:

- **Embedding Layer**: Converts tokenized Java code into dense vectors.
- **LSTM Layer**: A 128-unit LSTM layer captures sequential relationships in the code tokens. 
- **Dense Layer**: A fully connected layer with 64 units and ReLU activation. 
- **Dropout**: A 50% dropout rate to prevent overfitting. 
- **Output Layer**: A sigmoid-activated layer for binary classification (vulnerable vs. non-vulnerable). 

model = tf.keras.Sequential([ 

`    `tf.keras.layers.Embedding(input\_dim=vocab\_size, output\_dim=128, input\_length=maxlen), 

`    `tf.keras.layers.LSTM(128, return\_sequences=False), 

`    `tf.keras.layers.Dense(64, activation='relu'), 

`    `tf.keras.layers.Dropout(0.5), 

`    `tf.keras.layers.Dense(1, activation='sigmoid') 

]) 

The model was compiled using the **Adam optimizer** and **binary cross-entropy loss**, and trained for 5 epochs. 

![](Aspose.Words.b7c728c7-6e89-4ddd-88d9-5c4c736d461a.001.jpeg)

7. **Training and Evaluation** 
1. ***Training*** 

The LSTM model was trained on the converted and augmented Java dataset. Class weights were applied during training to handle class imbalance, and the model converged over 5 epochs, improving in accuracy. 

2. ***Evaluation*** 

The model’s performance was evaluated using metrics such as accuracy, precision, recall, and F1-score. The initial results indicated the model’s ability to classify Java code snippets as vulnerable or non-vulnerable. 

8. **Tool for Vulnerability Detection** 

We built a tool that predicts vulnerabilities in Java code snippets using the trained LSTM model. This tool: 

1. **Preprocesses** the input code snippet. 
1. **Tokenizes** and converts the code into a sequence of tokens.
1. **Classifies** the code as either vulnerable or non-vulnerable using the trained model.

def predict\_vulnerability(code\_snippet): 

`    `code\_snippet = preprocess\_code(code\_snippet) 

`    `tokenized\_code = ' '.join(tokenize\_java\_code(code\_snippet)) 

`    `seq = tokenizer.texts\_to\_sequences([tokenized\_code]) 

`    `padded\_seq = pad\_sequences(seq, padding='post', maxlen=maxlen)     prediction = model.predict(padded\_seq)[0][0] 

`    `return 'Vulnerable' if prediction > 0.5 else 'Not Vulnerable' 

9. **Challenges and Limitations** 
- **Dataset Conversion**: Some low-level vulnerabilities in C/C++ do not have direct equivalents in Java (e.g., manual memory management), making conversion challenging. 
- **Class Imbalance**: Vulnerable code snippets were less frequent, requiring careful handling of class weights during training. 
- **Complexity**: More advanced models could be explored for better performance.
10. **Results** 

The LSTM model successfully classified Java code snippets as vulnerable or non - vulnerable. With the converted dataset, the tool provided accurate results and can be a valuable addition to code review processes for Java developers.

11. **Output Screenshots** 

![](Aspose.Words.b7c728c7-6e89-4ddd-88d9-5c4c736d461a.002.jpeg)

![](Aspose.Words.b7c728c7-6e89-4ddd-88d9-5c4c736d461a.003.jpeg)

12. **Conclusion** 

Using **Tree-sitter**, we efficiently converted the C/C++ MegaVul dataset to Java, augmented it with Java-specific vulnerabilities, and trained an LSTM model for vulnerability detection. The resulting tool is capable of analyzing Java code and identifying vulnerabilities, providing significant support to developers in ensuring code security.

13. **Future Work** 
- **Advanced Models**: Future work could involve experimenting with Bi-LSTMs, GRUs, or transformer-based models like **CodeBERT** to improve accuracy. 
- **Dataset Expansion**: Adding more real-world Java vulnerabilities will improve the dataset and model generalization. 
- **Real-World Application**: Integration into CI/CD pipelines for real-time vulnerability detection in Java projects could enhance the tool’s practical utility.
14. **References** 
- Chao Ni et al. (2024). *MegaVul: A C/C++ Vulnerability Dataset with Comprehensive Code Representations*. MSR ’24 (2406.12415v1). 
