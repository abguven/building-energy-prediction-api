output "apprunner_service_url" {
  description = "Public URL of the App Runner service"
  value       = "https://${aws_apprunner_service.seattle.service_url}"
}
