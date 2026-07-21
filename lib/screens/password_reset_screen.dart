import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';

class PasswordResetScreen extends StatefulWidget {
  const PasswordResetScreen({super.key});

  @override
  State<PasswordResetScreen> createState() => _PasswordResetScreenState();
}

class _PasswordResetScreenState extends State<PasswordResetScreen> {
  final _emailController = TextEditingController();
  final _tokenController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  bool _requested = false;
  bool _submitting = false;

  @override
  void dispose() {
    _emailController.dispose();
    _tokenController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  Future<void> _requestReset() async {
    final email = _emailController.text.trim();
    if (email.isEmpty || !email.contains('@')) {
      _showError('Enter a valid email address.');
      return;
    }

    setState(() => _submitting = true);
    try {
      await Api().requestPasswordReset(email);
      if (!mounted) return;
      setState(() => _requested = true);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Check your email for a password reset link.'),
        ),
      );
    } catch (error) {
      _showError(error.toString());
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  Future<void> _confirmReset() async {
    final token = _tokenFromInput(_tokenController.text);
    final password = _passwordController.text;
    if (token.isEmpty) {
      _showError('Paste the reset token or reset link from your email.');
      return;
    }
    if (password.length < 8) {
      _showError('Your new password must contain at least 8 characters.');
      return;
    }
    if (password != _confirmPasswordController.text) {
      _showError('The passwords do not match.');
      return;
    }

    setState(() => _submitting = true);
    try {
      await Api().confirmPasswordReset(token, password);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Password updated. You can now sign in.')),
      );
      Navigator.pop(context);
    } catch (error) {
      _showError(error.toString());
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  String _tokenFromInput(String value) {
    final input = value.trim();
    final uri = Uri.tryParse(input);
    return uri?.queryParameters['token'] ?? input;
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red.shade700),
    );
  }

  InputDecoration _decoration(String label, IconData icon) {
    return InputDecoration(
      labelText: label,
      prefixIcon: Icon(icon, color: AppTheme.teal),
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.bg,
      appBar: AppBar(
        backgroundColor: AppTheme.bg,
        foregroundColor: AppTheme.textPrimary,
        elevation: 0,
        title: Text(
          'Reset password',
          style: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w700),
        ),
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 460),
            child: ListView(
              padding: const EdgeInsets.all(24),
              children: [
                Text(
                  _requested ? 'Choose a new password' : 'Find your account',
                  style: GoogleFonts.dmSans(
                    color: AppTheme.textPrimary,
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _requested
                      ? 'Paste the reset token or the complete link sent to your email.'
                      : 'Enter the email address you used to create your StayBuddy account.',
                  style: GoogleFonts.dmSans(
                    color: AppTheme.textMuted,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 28),
                if (!_requested) ...[
                  TextField(
                    controller: _emailController,
                    keyboardType: TextInputType.emailAddress,
                    decoration: _decoration(
                      'Email address',
                      Icons.mail_outline_rounded,
                    ),
                  ),
                  const SizedBox(height: 18),
                  _actionButton('Send reset link', _requestReset),
                ] else ...[
                  TextField(
                    controller: _tokenController,
                    autocorrect: false,
                    enableSuggestions: false,
                    decoration: _decoration(
                      'Reset token or link',
                      Icons.link_rounded,
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextField(
                    controller: _passwordController,
                    obscureText: true,
                    enableSuggestions: false,
                    autocorrect: false,
                    decoration: _decoration(
                      'New password',
                      Icons.lock_outline_rounded,
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextField(
                    controller: _confirmPasswordController,
                    obscureText: true,
                    enableSuggestions: false,
                    autocorrect: false,
                    decoration: _decoration(
                      'Confirm new password',
                      Icons.lock_reset_rounded,
                    ),
                  ),
                  const SizedBox(height: 18),
                  _actionButton('Update password', _confirmReset),
                  TextButton(
                    onPressed: _submitting
                        ? null
                        : () => setState(() => _requested = false),
                    child: const Text('Request another link'),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _actionButton(String label, Future<void> Function() action) {
    return SizedBox(
      height: 50,
      child: ElevatedButton(
        onPressed: _submitting ? null : action,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
        ),
        child: _submitting
            ? const SizedBox(
                height: 22,
                width: 22,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2,
                ),
              )
            : Text(
                label,
                style: GoogleFonts.dmSans(fontWeight: FontWeight.w600),
              ),
      ),
    );
  }
}
