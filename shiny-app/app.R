library(shiny)
library(AzureStor)

# Define the Azure Storage Account details
storage_account_name <- Sys.getenv("BLOB_ACCOUNT_NAME")
storage_account_key <- Sys.getenv("BLOB_ACCOUNT_KEY")
container_name <- Sys.getenv("BLOB_CONTAINER_NAME")

# Function to read the CSV file from Azure Blob Storage

read_azure_csv <- function() {
  bl_endp_key <- storage_endpoint(paste("https://", storage_account_name, ".blob.core.windows.net", sep = ""), key=storage_account_key)
  cont <- storage_container(bl_endp_key, "datahub")
  list_storage_files(cont)
  list_blob_containers(bl_endp_key)
  data <- storage_read_csv(cont, "sample.csv", stringsAsFactors=TRUE)
  return(data)
}

# UI
ui <- fluidPage(
  titlePanel("FSDH Azure Storage Data Presentation"),
  sidebarLayout(
    sidebarPanel(),
    mainPanel(
      tableOutput("data_table")
    )
  )
)

# Server
server <- function(input, output) {
  data <- reactive({
    read_azure_csv()
  })

  output$data_table <- renderTable({
    data()
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)