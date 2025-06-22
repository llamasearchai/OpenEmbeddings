## Description

Brief description of the changes in this PR.

Fixes #(issue number) <!-- If applicable -->

## Type of Change

Please delete options that are not relevant:

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test improvements
- [ ] CI/CD improvements

## Changes Made

### Core Changes
- [ ] Modified core embedding functionality
- [ ] Updated model interfaces
- [ ] Changed API endpoints
- [ ] Modified CLI commands

### Components Affected
- [ ] Dense Embeddings
- [ ] Sparse Embeddings
- [ ] Hybrid Retrieval
- [ ] Re-ranking
- [ ] CLI
- [ ] Web API
- [ ] Evaluation
- [ ] Quantization/Optimization
- [ ] Docker
- [ ] Documentation

## Testing

### Test Coverage
- [ ] Added tests for new functionality
- [ ] Updated existing tests
- [ ] All tests pass locally
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)

### Manual Testing
Please describe the manual testing you performed:

```bash
# Example commands used for testing
openembeddings --help
python -m pytest tests/
```

### Test Results
```
# Paste test output here if relevant
```

## Documentation

- [ ] Updated docstrings for new/modified functions
- [ ] Updated README.md (if needed)
- [ ] Updated CHANGELOG.md
- [ ] Added examples for new features
- [ ] Updated CLI help text
- [ ] Updated API documentation

## Performance Impact

- [ ] No performance impact
- [ ] Performance improvement (provide benchmarks)
- [ ] Potential performance regression (explain why acceptable)

### Benchmarks (if applicable)
```
# Before:
# After:
```

## Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes (list them below)

### Breaking Changes List
1. 
2. 
3. 

### Migration Guide
If there are breaking changes, provide a migration guide:

```python
# Old way:
# old_function()

# New way:
# new_function()
```

## Dependencies

- [ ] No new dependencies
- [ ] Added new required dependencies
- [ ] Added new optional dependencies
- [ ] Updated existing dependencies

### New Dependencies
List any new dependencies and why they're needed:
- `package-name>=1.0.0` - Used for X functionality

## Deployment Notes

- [ ] No special deployment considerations
- [ ] Requires environment variable changes
- [ ] Requires database migrations
- [ ] Requires configuration updates

### Special Instructions
Any special instructions for deployment:

## Screenshots/Examples

If applicable, add screenshots or code examples to help explain your changes:

```python
# Example usage of new feature
```

## Checklist

### Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is well-commented, particularly in hard-to-understand areas
- [ ] No debugging code left in the codebase
- [ ] No TODO comments left unaddressed

### Testing
- [ ] Tests are comprehensive and cover edge cases
- [ ] Tests are well-named and self-documenting
- [ ] Mock objects are used appropriately
- [ ] Tests run quickly and reliably

### Documentation
- [ ] All public APIs are documented
- [ ] Examples are provided for complex features
- [ ] Documentation is clear and accurate
- [ ] Links in documentation are working

### Security
- [ ] No sensitive information exposed
- [ ] Input validation is adequate
- [ ] Security implications have been considered
- [ ] No new security vulnerabilities introduced

### Accessibility
- [ ] CLI output is accessible
- [ ] Error messages are clear and helpful
- [ ] API responses are well-structured

## Review Notes

### Areas of Focus
Please pay special attention to:
- 
- 
- 

### Questions for Reviewers
- 
- 
- 

## Additional Context

Add any other context about the pull request here:

---

**Reviewer Guidelines:**
- Ensure code quality and consistency
- Verify test coverage is adequate
- Check for potential security issues
- Validate documentation is complete
- Test the changes locally if possible 