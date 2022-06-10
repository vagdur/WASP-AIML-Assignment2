library(shiny)
library(MASS)
library(e1071)

source("staticFunctions.r")

# Define UI for our application - it should essentially be a better version of a Jupyter notebook for
# presentation.
ui <- fluidPage(
    # Title of the page
    titlePanel("SVM, Kernels, and Non-linear Data", windowTitle = "WASP AIML Project"),
    withMathJax(),
    tags$p("In this assignment, we are going to explore how Support Vector Machines (SVMs) learn non-linear data."),
    tags$p("So, first off, let us generate some data! To make it more interesting, I have included the option of varying the number of points, and having them form an ellipse instead of just a sphere (by correlating their two coordinates), and having some label noise - so we can see what these factors do to the performance of our SVMs."),
    sidebarLayout(
      sidebarPanel(
        sliderInput("n","\\(n\\), the number of points:", min = 1, max = 1000, value = 200),
        sliderInput("rho","\\(\\rho\\), correlation between the coordinates of the points", min = -1, max = 1, value = 0, step = 0.01),
        "The points are drawn from a normal distribution with mean \\((0,0)\\) and correlation matrix \\(\\begin{pmatrix}
1 & \\rho \\\\
\\rho & 1 
\\end{pmatrix}\\).",
        sliderInput("epsilon", "\\(\\epsilon\\), label error probability", min = 0, max = 0.5, value = 0),
        "Each points gets an incorrect label with probability \\(\\epsilon\\)."
      ),
      mainPanel(
        plotOutput("dataPlot", width = "500px", height = "500px")
      )
    ),
    tags$p("So, now let us try some different SVM kernels and parameters and see what they do."),
    sidebarLayout(
      sidebarPanel(
        selectInput("kernel", "Type of kernel",
                    choices = c(Linear = "linear",
                                Polynomial = "polynomial",
                                Radial = "radial",
                                Sigmoid = "sigmoid")
                    ),
        uiOutput("kernelFormula"),
        conditionalPanel(
          condition = "input.kernel == 'polynomial'",
          sliderInput("degree","Polynomial degree, \\(d\\)", min = 2, max = 10, value = 2),
        ),
        conditionalPanel(
          condition = "input.kernel != 'linear'",
          sliderInput("gamma", "\\(\\gamma\\)", min = 0.2, max = 4, value = 1)
        ),
        conditionalPanel(
          condition = "input.kernel == 'polynomial' || input.kernel == 'sigmoid'",
          sliderInput("coef0", "\\(k\\)", min = 0, max = 10, value = 1)
        )
      ),
      mainPanel(
        plotOutput("svmPlot")
      )
    )
)

# Server logic specifying how the UI should work:
server <- function(input, output) {
  # We start by sampling the data that our SVM should be classifying, and computing its labels:
  x <- reactive(generateData(n = input$n,
                             rho = input$rho))
  labels <- reactive(computeDataLabels(x(), epsilon = input$epsilon))
  # Of course, a plot of the generated data is nice.
  # Since we want the plot to be square and centered around zero, we need to compute the right x- and y-lims of the plot first:
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
  
  # Having generated the data, we can now compute our SVM and show the results.
  # The user should get to see the form of the kernel they've picked, and putting
  # LaTeX in the dropdown menu turned out to not work in any easy way. So we display
  # it below the input box:
  output$kernelFormula <- renderUI({
    if (input$kernel == "linear") {
      withMathJax("$$
                  k(x,x') = \\langle x, x'\\rangle
                  $$")
    } else if (input$kernel == "polynomial") {
      withMathJax("$$
                  k(x,x') = \\left(\\gamma \\langle x, x' \\rangle + k\\right)^d
                  $$")
    } else if (input$kernel == "radial") {
      withMathJax("$$
                  k(x,x') = e^{-\\gamma \\left|x - x'\\right|^2}
                  $$")
    } else if (input$kernel == "sigmoid") {
      withMathJax("$$
                  k(x, x') = \\tanh\\left(\\gamma \\langle x, x'\\rangle + k\\right)
                  $$")
    }
  })
  # First, we package up the data and compute the SVM:
  svmData <- reactive(data.frame(X1 = x()[,1], X2 = x()[,2], Y = as.factor(labels())))
  fitSVM <- reactive(
    svm(
      Y ~ X1 + X2, # The abstract specification of the model
      data = svmData(),
      kernel = input$kernel,
      degree = input$degree,
      gamma = input$gamma,
      coef0 = input$coef0,
      scale = FALSE
    )
  )
  # Then, we are fortunate enough that the SVM package we are using comes with a built-in
  # function to plot an SVM, with all its data and the support vectors illustrated:
  output$svmPlot <- renderPlot(plot(fitSVM(), data = svmData()))
}

# Run the application 
shinyApp(ui = ui, server = server)
