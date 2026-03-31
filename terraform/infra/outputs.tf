output "ecr_repository_url" {
  description = "ECR repository URL — use this to tag and push your Docker image"
  value       = aws_ecr_repository.seattle.repository_url
}

output "apprunner_iam_role_arn" {
  description = "IAM role ARN for App Runner — used in terraform/service"
  value       = aws_iam_role.apprunner_ecr.arn
}
