import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../widgets/video_background.dart';

class OwnerFeesScreen extends StatefulWidget {
  final String hostelName;
  const OwnerFeesScreen({super.key, required this.hostelName});

  @override
  State<OwnerFeesScreen> createState() => _OwnerFeesScreenState();
}

class _OwnerFeesScreenState extends State<OwnerFeesScreen> {
  DateTime _month = DateTime(DateTime.now().year, DateTime.now().month);
  List<dynamic> _records = const [];
  bool _loading = true;

  String get _monthKey =>
      '${_month.year.toString().padLeft(4, '0')}-${_month.month.toString().padLeft(2, '0')}-01';
  String get _monthLabel =>
      const [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December',
      ][_month.month - 1] +
      ' ${_month.year}';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() => _loading = true);
    try {
      final records = await Api().getOwnerFeeRecords(_monthKey);
      if (mounted) setState(() => _records = records);
    } catch (error) {
      if (mounted)
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not load rent records: $error')),
        );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  num _number(dynamic value) =>
      value is num ? value : num.tryParse('$value') ?? 0;
  String _money(dynamic value) => 'PKR ${_number(value).toStringAsFixed(0)}';

  Future<void> _setup(Map<String, dynamic> record) async {
    try {
      await Api().setupOwnerRentRecord(
        bookingId: _number(record['booking_id']).toInt(),
        billingMonth: _monthKey,
      );
      await _load();
    } catch (error) {
      if (mounted)
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('$error')));
    }
  }

  Future<void> _cash(Map<String, dynamic> record) async {
    final amount = TextEditingController(
      text: _number(record['remaining_amount']).toStringAsFixed(0),
    );
    final note = TextEditingController();
    DateTime receivedOn = DateTime.now();
    final save = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: Text(
            'Record cash rent',
            style: GoogleFonts.dmSans(fontWeight: FontWeight.w800),
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: amount,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(labelText: 'Amount (PKR)'),
              ),
              TextField(
                controller: note,
                decoration: const InputDecoration(labelText: 'Note (optional)'),
              ),
              const SizedBox(height: 12),
              TextButton.icon(
                onPressed: () async {
                  final picked = await showDatePicker(
                    context: context,
                    initialDate: receivedOn,
                    firstDate: DateTime(2020),
                    lastDate: DateTime(2100),
                  );
                  if (picked != null) setDialogState(() => receivedOn = picked);
                },
                icon: const Icon(Icons.calendar_today_outlined),
                label: Text(
                  '${receivedOn.day}/${receivedOn.month}/${receivedOn.year}',
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext, false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(dialogContext, true),
              child: const Text('Mark paid'),
            ),
          ],
        ),
      ),
    );
    if (save != true) return;
    try {
      await Api().recordOwnerCashFeePayment(
        invoiceId: _number(record['invoice_id']).toInt(),
        amount: double.parse(amount.text),
        receivedOn:
            '${receivedOn.year.toString().padLeft(4, '0')}-${receivedOn.month.toString().padLeft(2, '0')}-${receivedOn.day.toString().padLeft(2, '0')}',
        note: note.text,
      );
      await _load();
    } catch (error) {
      if (mounted)
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Could not record payment: $error')),
        );
    }
  }

  @override
  Widget build(BuildContext context) => ResponsiveShell(
    child: Scaffold(
      backgroundColor: Colors.transparent,
      body: VideoBackground(
        assetPath: 'assets/images/background.mp4',
        overlayOpacity: .52,
        cardMargin: const EdgeInsets.all(14),
        cardRadius: 32,
        child: SafeArea(
          child: RefreshIndicator(
            onRefresh: _load,
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(
                        Icons.arrow_back_rounded,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Fees & Rent',
                        style: GoogleFonts.playfairDisplay(
                          color: Colors.white,
                          fontWeight: FontWeight.w700,
                          fontSize: 29,
                        ),
                      ),
                    ),
                    IconButton(
                      onPressed: _load,
                      icon: const Icon(
                        Icons.refresh_rounded,
                        color: AppTheme.tealMint,
                      ),
                    ),
                  ],
                ),
                Text(
                  widget.hostelName,
                  style: GoogleFonts.dmSans(
                    color: Colors.white70,
                    fontSize: 11,
                  ),
                ),
                const SizedBox(height: 22),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: const Color(0xE609292B),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: Row(
                    children: [
                      IconButton(
                        onPressed: () {
                          setState(
                            () => _month = DateTime(
                              _month.year,
                              _month.month - 1,
                            ),
                          );
                          _load();
                        },
                        icon: const Icon(
                          Icons.chevron_left_rounded,
                          color: Colors.white,
                        ),
                      ),
                      Expanded(
                        child: Column(
                          children: [
                            Text(
                              _monthLabel,
                              style: GoogleFonts.dmSans(
                                color: Colors.white,
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                            Text(
                              'Monthly student rent register',
                              style: GoogleFonts.dmSans(
                                color: Colors.white54,
                                fontSize: 9,
                              ),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        onPressed: () {
                          setState(
                            () => _month = DateTime(
                              _month.year,
                              _month.month + 1,
                            ),
                          );
                          _load();
                        },
                        icon: const Icon(
                          Icons.chevron_right_rounded,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 18),
                if (_loading)
                  const Center(
                    child: Padding(
                      padding: EdgeInsets.all(40),
                      child: CircularProgressIndicator(
                        color: AppTheme.tealMint,
                      ),
                    ),
                  )
                else if (_records.isEmpty)
                  Padding(
                    padding: const EdgeInsets.all(30),
                    child: Text(
                      'No registered students found for this hostel.',
                      textAlign: TextAlign.center,
                      style: GoogleFonts.dmSans(color: Colors.white70),
                    ),
                  )
                else
                  ..._records.map(
                    (raw) => _recordCard(Map<String, dynamic>.from(raw as Map)),
                  ),
              ],
            ),
          ),
        ),
      ),
    ),
  );

  Widget _recordCard(Map<String, dynamic> record) {
    final invoice = record['invoice_id'] != null;
    final remaining = _number(record['remaining_amount']);
    final paid = _number(record['paid_amount']);
    final fullyPaid = invoice && remaining <= 0;
    final color = fullyPaid ? AppTheme.tealMint : const Color(0xFFFFC978);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: const Color(0xE609292B),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              CircleAvatar(
                backgroundColor: AppTheme.teal.withOpacity(.25),
                child: Text(
                  (record['student_name']?.toString().isNotEmpty ?? false)
                      ? record['student_name'].toString()[0].toUpperCase()
                      : '?',
                  style: const TextStyle(color: AppTheme.tealMint),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      record['student_name']?.toString() ?? 'Student',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    Text(
                      '${record['floor_label'] ?? '-'} · Room ${record['room_number'] ?? '-'} · Bed ${record['bed_number'] ?? '-'}',
                      style: GoogleFonts.dmSans(
                        color: Colors.white60,
                        fontSize: 9,
                      ),
                    ),
                  ],
                ),
              ),
              Chip(
                label: Text(
                  fullyPaid
                      ? 'Paid'
                      : invoice
                      ? 'Due'
                      : 'Not set up',
                ),
                backgroundColor: color.withOpacity(.16),
                labelStyle: GoogleFonts.dmSans(
                  color: color,
                  fontSize: 9,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Divider(color: Colors.white.withOpacity(.08)),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Rent: ${_money(invoice ? record['invoice_amount'] : record['room_monthly_rent'])}',
                style: GoogleFonts.dmSans(color: Colors.white70, fontSize: 10),
              ),
              Text(
                'Paid: ${_money(paid)}',
                style: GoogleFonts.dmSans(
                  color: AppTheme.tealMint,
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: !invoice
                  ? () => _setup(record)
                  : fullyPaid
                  ? null
                  : () => _cash(record),
              icon: Icon(
                !invoice
                    ? Icons.receipt_long_outlined
                    : Icons.payments_outlined,
                size: 16,
              ),
              label: Text(
                !invoice
                    ? 'Create rent record'
                    : fullyPaid
                    ? 'Rent paid'
                    : 'Mark cash payment',
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.teal,
                foregroundColor: Colors.white,
                disabledBackgroundColor: Colors.white12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
