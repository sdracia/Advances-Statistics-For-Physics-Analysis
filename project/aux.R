library(tm)
library(textstem)
library(SnowballC)
library(dplyr)
library(tidyr)

lemmatize_text <- function(text) {
  lemmatized <- textstem::lemmatize_words(unlist(strsplit(text, "\\s+")))
  lemmatized <- SnowballC::wordStem(lemmatized, language = "en")

  return(paste(lemmatized, collapse = " "))
}

remove_numbers_inside_words <- function(text) {
  words <- unlist(strsplit(text, "\\s+"))

  clean_words <- lapply(words, function(word) {
    if (grepl("\\d", word)) {  # Check if the word contains digits
      word <- gsub("\\d", "", word)  # Remove digits
    }
    return(word)
  })

  cleaned_text <- paste(clean_words, collapse = " ")
  return(cleaned_text)
}

clean <- function(document, tokenize = TRUE, lemmatize = TRUE){
  clean_doc <- tm::VCorpus(tm::VectorSource(document))

  if (tokenize) {
    clean_doc <- tm::tm_map(clean_doc, tm::content_transformer(tolower))
    clean_doc <- tm::tm_map(clean_doc, tm::removePunctuation)
    clean_doc <- tm::tm_map(clean_doc, tm::removeWords, tm::stopwords("en"))
    clean_doc <- tm::tm_map(clean_doc, tm::removeNumbers)
    clean_doc <- tm::tm_map(clean_doc, tm::content_transformer(remove_numbers_inside_words))
    clean_doc <- tm::tm_map(clean_doc, tm::stripWhitespace)
  }

  if (lemmatize) {
    clean_doc <- tm::tm_map(clean_doc, tm::content_transformer(lemmatize_text))
  }

  return(sapply(clean_doc, NLP::content))
}

# other possibilities for cleaning:
# 1) eliminate terms that appear less than a chosen threshold
# 2) compute the TF-IDF score

get_vocabulary <- function(document) {
  words <- unlist(strsplit(document, "\\s+"))
  words <- words[words != ""]
  words_table <- table(words)

  words_freq <- as.data.frame(words_table, stringsAsFactors = FALSE)
  colnames(words_freq) <- c("word", "occurrencies")

  vocabulary <- words_freq[words_freq$occurrencies > 5, ]$word
  return(vocabulary)
}

clean_empty_rows <- function(dataframe) {
  empty_rows <- which(nchar(trimws(dataframe$Text)) == 0)
  if (length(empty_rows) != 0) {
    dataframe <- dataframe[-empty_rows, ]
  }
  return(dataframe)
}

counter <- function(document, term) {
  return(sum(unlist(strsplit(document, "\\s+")) == term) + 1)
}


train_multinomial_nb <- function(classes, data) {
  vocabulary <- get_vocabulary(data$Text)
  num_docs <- nrow(data)
  
  # Calculate priors
  prior <- table(data$Label) / num_docs
  prior <- as.numeric(prior[match(classes, names(prior))])
  
  # Create a data frame for text and labels
  df <- data %>%
    group_by(Label) %>%
    summarize(Text = paste(Text, collapse = " "), .groups = 'drop')
  
  # Tokenize the text and create a term-document matrix
  term_matrix <- df %>%
    mutate(Text = strsplit(Text, "\\s+")) %>%
    unnest(Text) %>%
    filter(Text != "") %>%  # Remove empty strings
    count(Label, Text) %>%
    pivot_wider(names_from = Text, values_from = n, values_fill = list(n = 0)) %>%
    as.data.frame()
  
  rownames(term_matrix) <- term_matrix$Label
  term_matrix <- term_matrix[, -1]  # Remove the Label column
  
  # Ensure vocabulary alignment
  term_matrix <- term_matrix %>%
    select(any_of(vocabulary)) %>%
    mutate(across(everything(), ~ . + 1)) # Laplace smoothing
  
  # Sum term frequencies for each class
  term_sums <- rowSums(term_matrix)
  
  # Calculate posterior probabilities
  post <- sweep(term_matrix, 1, term_sums, FUN = "/")
  
  return(list(vocab = vocabulary, prior = prior, post = as.matrix(post)))
}
