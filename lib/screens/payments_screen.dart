import 'package:flutter/material.dart';
import 'package:flutter_stripe/flutter_stripe.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';

class PaymentsScreen extends StatefulWidget {
  final Map<String, dynamic>? booking;

  const PaymentsScreen({super.key, this.booking});

  @override
  State<PaymentsScreen> createState() => _PaymentsScreenState();
}

class _PaymentsScreenState extends State<PaymentsScreen> {
  static const _stripePublishableKey = String.fromEnvironment(
    'STRIPE_PUBLISHABLE_KEY',
  );

  final Api _api = Api();
  List<dynamic> _payments = const [];
  bool _isLoading = true;
  bool _isStartingPayment = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadPayments();
  }

  Future<void> _loadPayments() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });
    try {
      final payments = await _api.getPayments();
      if (!mounted) return;
      setState(() {
        _payments = payments;
        _isLoading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
        _error = error.toString();
      });
    }
  }

  int? get _bookingId =>
      int.tryParse(widget.booking?['rawId'].toString() ?? '');

  List<dynamic> get _bookingPayments {
    final bookingId = _bookingId;
    if (bookingId == null) return const [];
    return _payments
        .where(
          (payment) =>
              int.tryParse(payment['booking_id'].toString()) == bookingId,
        )
        .toList();
  }

  bool get _canStartPayment {
    if (_bookingId == null || widget.booking?['status'] != 'Confirmed') {
      return false;
    }
    return !_bookingPayments.any((payment) {
      final status = payment['status']?.toString().toLowerCase();
      return status == 'pending' || status == 'succeeded';
    });
  }

  Future<void> _startPayment() async {
    final bookingId = _bookingId;
    if (bookingId == null || !_canStartPayment || _isStartingPayment) return;
    if (_stripePublishableKey.isEmpty) {
      _showMessage(
        'Secure payments are not configured in this app build.',
        isError: true,
      );
      return;
    }

    setState(() => _isStartingPayment = true);
    try {
      final result = await _api.createPaymentIntent(bookingId);
      final clientSecret = result['client_secret']?.toString();
      if (clientSecret == null || clientSecret.isEmpty) {
        throw Exception(
          'The payment provider did not return a payment session.',
        );
      }
      await Stripe.instance.initPaymentSheet(
        paymentSheetParameters: SetupPaymentSheetParameters(
          merchantDisplayName: 'StayBuddy',
          paymentIntentClientSecret: clientSecret,
          style: ThemeMode.light,
        ),
      );
      await Stripe.instance.presentPaymentSheet();
      if (!mounted) return;
      _showMessage(
        'Payment submitted. Its status will update after secure confirmation.',
      );
      await _loadPayments();
    } on StripeException catch (error) {
      if (!mounted) return;
      _showMessage(
        error.error.localizedMessage ?? 'Payment was not completed.',
        isError: true,
      );
      await _loadPayments();
    } catch (error) {
      if (!mounted) return;
      _showMessage(error.toString(), isError: true);
      await _loadPayments();
    } finally {
      if (mounted) setState(() => _isStartingPayment = false);
    }
  }

  void _showMessage(String message, {bool isError = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, style: GoogleFonts.dmSans()),
        backgroundColor: isError ? Colors.red.shade700 : AppTheme.tealDeep,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  String _formatAmount(dynamic amount, dynamic currency) {
    final value = num.tryParse(amount?.toString() ?? '')?.round();
    if (value == null) return 'Amount unavailable';
    final formatted = value.toString().replaceAllMapped(
      RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
      (match) => '${match[1]},',
    );
    return '${currency?.toString().toUpperCase() ?? 'PKR'} $formatted';
  }

  String _formatDate(dynamic value) {
    final date = DateTime.tryParse(value?.toString() ?? '')?.toLocal();
    if (date == null) return '';
    final minute = date.minute.toString().padLeft(2, '0');
    final hour = date.hour % 12 == 0 ? 12 : date.hour % 12;
    return '${date.day}/${date.month}/${date.year} at $hour:$minute ${date.hour >= 12 ? 'PM' : 'AM'}';
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'succeeded':
        return AppTheme.green;
      case 'pending':
        return Colors.orange.shade700;
      case 'failed':
      case 'cancelled':
        return Colors.red.shade700;
      case 'refunded':
        return const Color(0xFF185FA5);
      default:
        return AppTheme.textMuted;
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: AppBar(
          backgroundColor: AppTheme.tealDeep,
          foregroundColor: Colors.white,
          title: Text(
            'Payments',
            style: GoogleFonts.playfairDisplay(fontWeight: FontWeight.w700),
          ),
        ),
        body: RefreshIndicator(
          color: AppTheme.teal,
          onRefresh: _loadPayments,
          child: _buildBody(),
        ),
      ),
    );
  }

  Widget _buildBody() {
    if (_isLoading) {
      return const Center(
        child: CircularProgressIndicator(color: AppTheme.teal),
      );
    }
    if (_error != null) {
      return _buildError();
    }

    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
      children: [
        if (widget.booking != null) ...[
          _buildBookingPaymentCard(),
          const SizedBox(height: 20),
          Text(
            'Payment history',
            style: GoogleFonts.dmSans(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 8),
        ],
        if (_payments.isEmpty)
          _buildEmpty()
        else
          ..._payments.map(
            (payment) => Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: _buildPaymentRow(
                Map<String, dynamic>.from(payment as Map),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildBookingPaymentCard() {
    final status = widget.booking?['status']?.toString() ?? '';
    final hasPendingPayment = _bookingPayments.any(
      (payment) => payment['status']?.toString().toLowerCase() == 'pending',
    );
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            widget.booking?['hostel']?.toString() ?? 'Booking payment',
            style: GoogleFonts.dmSans(
              fontSize: 16,
              fontWeight: FontWeight.w700,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Booking ${widget.booking?['id'] ?? ''}',
            style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
          ),
          const SizedBox(height: 14),
          if (status != 'Confirmed')
            Text(
              'Payments become available once your booking is confirmed.',
              style: GoogleFonts.dmSans(
                fontSize: 12,
                color: AppTheme.textMuted,
              ),
            )
          else if (hasPendingPayment)
            Text(
              'A payment is awaiting secure confirmation. Refresh to check for an updated status.',
              style: GoogleFonts.dmSans(
                fontSize: 12,
                color: Colors.orange.shade800,
              ),
            )
          else if (_canStartPayment)
            SizedBox(
              width: double.infinity,
              child: FilledButton.icon(
                onPressed: _isStartingPayment ? null : _startPayment,
                icon: _isStartingPayment
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.lock_outline_rounded),
                label: Text(
                  _isStartingPayment
                      ? 'Opening secure payment...'
                      : 'Pay securely',
                  style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                ),
              ),
            ),
          if (_stripePublishableKey.isEmpty && _canStartPayment) ...[
            const SizedBox(height: 10),
            Text(
              'This build needs a Stripe publishable key before secure checkout can open.',
              style: GoogleFonts.dmSans(
                fontSize: 11,
                color: AppTheme.textMuted,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildPaymentRow(Map<String, dynamic> payment) {
    final status = payment['status']?.toString().toLowerCase() ?? 'pending';
    final color = _statusColor(status);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppTheme.border),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.13),
              shape: BoxShape.circle,
            ),
            child: Icon(
              status == 'succeeded'
                  ? Icons.check_circle_outline_rounded
                  : Icons.receipt_long_rounded,
              color: color,
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _formatAmount(payment['amount'], payment['currency']),
                  style: GoogleFonts.dmSans(
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  'Booking #${payment['booking_id']} · ${_formatDate(payment['created_at'])}',
                  style: GoogleFonts.dmSans(
                    fontSize: 11,
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              status[0].toUpperCase() + status.substring(1),
              style: GoogleFonts.dmSans(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmpty() => Padding(
    padding: const EdgeInsets.only(top: 96),
    child: Column(
      children: [
        Icon(Icons.receipt_long_outlined, size: 52, color: AppTheme.textMuted),
        const SizedBox(height: 12),
        Text(
          'No payment records yet',
          style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 6),
        Text(
          'Confirmed booking payments will appear here.',
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
        ),
      ],
    ),
  );

  Widget _buildError() => ListView(
    padding: const EdgeInsets.all(24),
    children: [
      const SizedBox(height: 120),
      Icon(Icons.error_outline_rounded, size: 42, color: Colors.red.shade400),
      const SizedBox(height: 12),
      Text(
        'Payments could not be loaded.',
        textAlign: TextAlign.center,
        style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
      ),
      const SizedBox(height: 8),
      Text(
        _error!,
        textAlign: TextAlign.center,
        style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
      ),
      const SizedBox(height: 16),
      Center(
        child: FilledButton.icon(
          onPressed: _loadPayments,
          icon: const Icon(Icons.refresh_rounded),
          label: const Text('Retry'),
        ),
      ),
    ],
  );
}
