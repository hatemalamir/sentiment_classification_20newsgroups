# import required libraries
library(tm)
library(randomForest)
library(rpart)

# reading 100 training documents from the first subject, rec.autos
rec.autos.train.source <- DirSource("input/20news-bydate-train/rec.autos")
rec.autos.train <- Corpus(URISource(rec.autos.train.source$filelist), readerControl=list(reader=readPlain))

# reading 100 test documents from the first subject, rec.autos
rec.autos.test.source <- DirSource("input/20news-bydate-test/rec.autos")
rec.autos.test <- Corpus(URISource(rec.autos.train.source$filelist), readerControl=list(reader=readPlain))

# reading 100 training documents from the second subject, sci.space
sci.space.train.source <- DirSource("input/20news-bydate-train/sci.space")
sci.space.train <- Corpus(URISource(sci.space.train.source$filelist), readerControl=list(reader=readPlain))

# reading 100 test documents from the first subject, sci.space
sci.space.test.source <- DirSource("input/20news-bydate-test/sci.space")
sci.space.test <- Corpus(URISource(sci.space.train.source$filelist[1:100]), readerControl=list(reader=readPlain))

# combined corpus
corpus_all <- c(sci.space.train, rec.autos.train, sci.space.test, rec.autos.test)

#>>> data preparation...start

# convert texts to lowercase
corpus_clean <- tm_map(corpus_all, content_transformer(tolower))

# remove numbers. They are usually unique per topic
corpus_clean <- tm_map(corpus_clean, removeNumbers)

# replace punctuations with a space
replacePunctuation <- function(x) {
  gsub("[[:punct:]]+", " ", x)
}
corpus_clean <-  tm_map(corpus_clean, content_transformer(replacePunctuation))

# remove stop words. They are not distiguishing 
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())

# stem documents
corpus_clean <- tm_map(corpus_clean, stemDocument)

# remove extra whitespaces
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

#<<< data preparation...done

# creating a document-term matrix
news_dtm <- as.matrix(DocumentTermMatrix(corpus_clean, control = list(bounds = list(global = c(5, Inf)))))

# creating train and test datasets
train_x <- news_dtm[1:(length(sci.space.train)+length(rec.autos.train)),]
test_x <- news_dtm[(nrow(train_x)+1):nrow(news_dtm),]

# create tags (labels)
train_y <- factor(c(rep("Sci", length(sci.space.train)), rep("Rec", length(rec.autos.train))))
test_y <- factor(c(rep("Sci", length(sci.space.test)), rep("Rec", length(rec.autos.test))))

train_df <- as.data.frame(train_x)
train_df$train_y <- train_y

test_df <- as.data.frame(test_x)
test_df$test_y <- test_y

# build a Random Forest on the data
news_rf <- randomForest(x = train_x, y = train_y)

# make RF predictions and measure accuracy
news_rf_pred <- predict(news_rf, test_x)
news_rf_table <- table(obsereved = test_y, predicted = news_rf_pred)

# build a Decision Tree on the data
news_tree <- rpart(train_y ~ ., data = train_df)

# make RF predictions and measure accuracy
news_tree_pred <- predict(news_tree, test_df, type = 'class')
news_tree_table <- table(obsereved = test_df$test_y, predicted = news_tree_pred)
