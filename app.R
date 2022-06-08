library(shiny)
library(MASS)

# Define UI for our application - it should essentially be a better version of a Jupyter notebook for
# presentation.
ui <- fluidPage(
    # Title of the page
    titlePanel("SVM, Kernels, and Non-linear Data", windowTitle = "WASP AIML Project"),
    "A bunch of content will go here!"
)

# Server logic specifying how the UI should work:
server <- function(input, output) {
  
}

# Run the application 
shinyApp(ui = ui, server = server)
