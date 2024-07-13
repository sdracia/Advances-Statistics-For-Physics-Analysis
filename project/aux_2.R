library(tm)
library(textstem)
library(SnowballC)
library(dplyr)

lemmatize_text <- function(text) {
  lemmatized <- textstem::lemmatize_words(unlist(strsplit(text, "\\s+")))
  lemmatized <- SnowballC::wordStem(lemmatized, language = "en")

  return(paste(lemmatized, collapse = " "))
}

filter_non_english_words <- function(text) {
  tokens <- unlist(strsplit(text, "\\s+"))
  is_english <- hunspell::hunspell_check(tokens)
  english_tokens <- tokens[is_english]
  cleaned_text <- paste(english_tokens, collapse = " ")
  return(cleaned_text)
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

to_space <- tm::content_transformer(function(x, pattern) {
  return(gsub(pattern, " ", x))
})

clean <- function(document, tokenize = TRUE, lemmatize = TRUE) {
  clean_doc <- tm::VCorpus(tm::VectorSource(document))

  if (tokenize) {
    clean_doc <- tm::tm_map(clean_doc, tm::content_transformer(tolower))
    clean_doc <- tm::tm_map(clean_doc, tm::removePunctuation)
    clean_doc <- tm::tm_map(clean_doc, tm::removeWords, tm::stopwords("en"))
    clean_doc <- tm::tm_map(clean_doc, tm::content_transformer(filter_non_english_words))
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

get_vocabulary <- function(document, threshold) {
  words <- unlist(strsplit(document, "\\s+"))
  words <- words[words != ""]
  words_table <- table(words)

  words_freq <- as.data.frame(words_table, stringsAsFactors = FALSE)
  colnames(words_freq) <- c("word", "occurrencies")

  vocabulary <- words_freq[words_freq$occurrencies >= threshold, ]$word
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


train_multinomial_nb <- function(classes, data, threshold) {
  n <- length(data$Text)
  vocabulary <- get_vocabulary(paste(data$Text, collapse = " "), threshold)

  prior <- numeric(length(classes))
  names(prior) <- classes
  post <- matrix(0, nrow = length(vocabulary), ncol = length(classes), dimnames = list(vocabulary, classes))

  for (c in seq_along(classes)) {
    class_label <- classes[c]
    docs_in_class <- data[data$Label == class_label, "Text"]
    prior[c] <- length(docs_in_class) / n

    textc <- paste(docs_in_class, collapse = " ")
    tokens <- table(strsplit(tolower(textc), "\\W+")[[1]])
    vocab_counts <- sapply(vocabulary, function(t) if (t %in% names(tokens)) tokens[t] else 0)

    post[, c] <- (vocab_counts + 1) / (sum(vocab_counts) + length(vocabulary))
  }

  return(list(vocab = vocabulary, prior = prior, condprob = post))
}

apply_multinomial_nb <- function(classes, vocab, prior, condprob, doc) {
  tokens <- intersect(unlist(strsplit(doc, "\\s+")), vocab)

  score_matrix <- matrix(0, nrow = length(tokens), ncol = length(classes))
  rownames(score_matrix) <- tokens
  colnames(score_matrix) <- classes

  for (c in seq_along(classes)) {
    for (t in seq_along(tokens)) {
      term <- tokens[t]
      score_matrix[t, c] <- log(condprob[term, c])
    }
  }

  # Add the log of prior probabilities to each class's score
  scores <- colSums(score_matrix) + log(prior)

  # Return the class with the highest score
  return(names(which.max(scores)))
}

get_vocabulary_two_label <- function(document, threshold) {
  words <- unlist(strsplit(document, "\\s+"))
  words <- words[words != ""]
  words_table <- table(words)

  words_freq <- as.data.frame(words_table, stringsAsFactors = FALSE)
  colnames(words_freq) <- c("word", "occurrencies")

  total_words <- sum(words_freq$occurrencies)
  print(nrow(words_freq))
  words_freq$occurrencies <- words_freq$occurrencies /total_words

  print(mean(words_freq$occurrencies))
  print(sd(words_freq$occurrencies))

  vocabulary <- words_freq[words_freq$occurrencies >= threshold, ]$word
  return(list(voc = vocabulary, df = words_freq))
}

vocabulary_tags <- function(df) {  
  tag_texts <- list()

  # Trova tutti i tag distinti
  all_tags <- unique(unlist(strsplit(df$Text_Tag, ",")))

  # Per ciascun tag, raccogli i testi associati
  for (tag in all_tags) {
    # Seleziona i documenti che hanno il tag specifico
    matching_docs <- df[grep(tag, df$Text_Tag), "Text"]

    # Unisci i testi dei documenti trovati e aggiungi alla lista dei tag_texts
    doc <- paste(matching_docs, collapse = " ")

    voc <- get_vocabulary_two_label(doc, 5)

    tag_texts <- append(tag_texts, voc)


  }

  tag_texts <- unlist(tag_texts)
}

train_multinomial_nb_new_two_label <- function(classes, data, threshold) {
  n <- length(data$Text)
  vocabulary <- get_vocabulary_two_label(paste(data$Text, collapse = " "), threshold)$voc

  prior <- numeric(length(classes))
  names(prior) <- classes
  post <- matrix(0, nrow = length(vocabulary), ncol = length(classes), dimnames = list(vocabulary, classes))

  for (c in seq_along(classes)) {
    class_label <- classes[c]
    docs_in_class <- data[data$Label == class_label, "Text"]
    prior[c] <- length(docs_in_class) / n

    textc <- paste(docs_in_class, collapse = " ")
    tokens <- table(strsplit(tolower(textc), "\\W+")[[1]])
    vocab_counts <- sapply(vocabulary, function(t) if (t %in% names(tokens)) tokens[t] else 0)

    post[, c] <- (vocab_counts + 1) / (sum(vocab_counts) + length(vocabulary))
  }

  return(list(vocab = vocabulary, prior = prior, condprob = post))
}