library(shiny)
library(MASS)

source("staticFunctions.r")

# Define UI for our application - it should essentially be a better version of a Jupyter notebook for
# presentation.
ui <- fluidPage(
    # Title of the page
    titlePanel("SVM, Kernels, and Non-linear Data", windowTitle = "WASP AIML Project"),
    tags$p("In this assignment, we are going to explore how Support Vector Machines (SVMs) learn non-linear data."),
    tags$p("So, first off, let us generate some data! To make it more interesting, I have included the option of varying the number of points, and having them form an ellipse instead of just a sphere (by correlating their two coordinates), and having some label noise - so we can see what these factors do to the performance of our SVMs."),
    sidebarLayout(
      sidebarPanel(
        sliderInput("n","Number of points:", min = 1, max = 1000, value = 200),
        sliderInput("rho","ρ, correlation between the coordinates of the points", min = 0, max = 1, value = 0),
        "The points are drawn from a normal distribution with mean (0,0) and correlation matrix ((1,ρ),(ρ,1)).",
        sliderInput("epsilon", "ε, label error probability", min = 0, max = 0.5, value = 0),
        "Each points gets an incorrect label with probability ε."
      ),
      mainPanel(
        plotOutput("dataPlot", width = "500px", height = "500px")
      )
    )
)

# Server logic specifying how the UI should work:
server <- function(input, output) {
  # We start by sampling the data that our SVM should be classifying, and computing its labels:
  x <- reactive(generateData(n = input$n,
                             rho = input$rho))
  labels <- reactive(as.integer(computeDataLabels(x(), epsilon = input$epsilon)))
  # Of course, a plot of the generated data is nice.
  # We of course want the plot to be square and centered around zero, so we need to compute the right x- and y-lims of the plot first:
  dataPlotXYLim <- reactive(max(x()[,1],x()[,2],-(x()[,1]),-(x()[,2])) + 0.1)
  output$dataPlot <- renderPlot({
    plot(
      x()[,1], x()[,2],
      col = labels() + 1,
      pch = 18,
      xlim = c(-dataPlotXYLim(),dataPlotXYLim()),
      ylim = c(-dataPlotXYLim(),dataPlotXYLim()),
      xlab = "",
      ylab = "",
      main = "Data our SVM should learn from"
    )
    abline(h = 0)
    abline(v = 0)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
