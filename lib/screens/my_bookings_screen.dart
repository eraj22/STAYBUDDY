import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'student_hostel_access_screen.dart';

class MyBookingsScreen extends StatefulWidget {
  const MyBookingsScreen({super.key});

  @override
  State<MyBookingsScreen> createState() => _MyBookingsScreenState();
}

class _MyBookingsScreenState extends State<MyBookingsScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tabs;

  final List<Map<String, dynamic>> _bookings = [];

  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 4, vsync: this);
    _loadBookings();
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  Future<void> _loadBookings() async {
    if (mounted) {
      setState(() {
        _loading = true;
        _error = null;
      });
    }

    try {
      final rows = await Api().getBookings();

      final mapped = rows
          .whereType<Map>()
          .map((row) => _mapBooking(Map<String, dynamic>.from(row)))
          .toList();

      if (!mounted) return;

      setState(() {
        _bookings
          ..clear()
          ..addAll(mapped);
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _loading = false;
        _error = e.toString();
      });
    }
  }

  Map<String, dynamic> _mapBooking(Map<String, dynamic> raw) {
    final statusRaw =
        (raw['status'] ?? 'pending').toString().trim().toLowerCase();

    final locationParts = <String>[
      if ((raw['hostel_area'] ?? '').toString().trim().isNotEmpty)
        raw['hostel_area'].toString().trim(),
      if ((raw['hostel_city'] ?? '').toString().trim().isNotEmpty)
        raw['hostel_city'].toString().trim(),
    ];

    final roomId = raw['room_id'];
    final price = _toInt(raw['single_room_price']);

    return {
      'id': _toInt(raw['id']),
      'hostelId': _toInt(raw['hostel_id']),
      'hostel': (raw['hostel_name'] ?? 'Hostel').toString(),
      'city': locationParts.isEmpty ? 'Location not available' : locationParts.join(' · '),
      'room': roomId == null ? 'Not assigned' : 'Room #$roomId',
      'checkIn': _formatDate(raw['check_in']),
      'checkOut': _formatDate(raw['check_out']),
      'duration': _durationText(raw['check_in'], raw['check_out']),
      'total': price,
      'statusRaw': statusRaw,
      'status': _displayStatus(statusRaw),
      'createdAt': raw['created_at'],
    };
  }

  int _toInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '') ?? 0;
  }

  String _displayStatus(String status) {
    switch (status) {
      case 'confirmed':
        return 'Confirmed';
      case 'completed':
        return 'Completed';
      case 'cancelled':
        return 'Cancelled';
      case 'pending':
      default:
        return 'Pending';
    }
  }

  String _formatDate(dynamic value) {
    if (value == null) return '-';

    final parsed = DateTime.tryParse(value.toString());
    if (parsed == null) return value.toString();

    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];

    return '${parsed.day.toString().padLeft(2, '0')} ${months[parsed.month - 1]} ${parsed.year}';
  }

  String _durationText(dynamic checkIn, dynamic checkOut) {
    final start = DateTime.tryParse(checkIn?.toString() ?? '');
    final end = DateTime.tryParse(checkOut?.toString() ?? '');

    if (start == null || end == null) return '-';

    final days = end.difference(start).inDays;

    if (days <= 0) return '-';
    if (days % 30 == 0) {
      final months = days ~/ 30;
      return '$months ${months == 1 ? 'month' : 'months'}';
    }

    return '$days days';
  }

  List<Map<String, dynamic>> _filtered(String status) {
    if (status == 'All') return _bookings;

    if (status == 'Past') {
      return _bookings
          .where((b) =>
              b['statusRaw'] == 'completed' || b['statusRaw'] == 'cancelled')
          .toList();
    }

    final wanted = status.toLowerCase();
    return _bookings.where((b) => b['statusRaw'] == wanted).toList();
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'Confirmed':
        return AppTheme.green;
      case 'Completed':
        return const Color(0xFF185FA5);
      case 'Cancelled':
        return Colors.red.shade600;
      case 'Pending':
      default:
        return Colors.orange.shade700;
    }
  }

  Color _statusBgColor(String status) {
    switch (status) {
      case 'Confirmed':
        return AppTheme.greenLight;
      case 'Completed':
        return const Color(0xFFE3EEF9);
      case 'Cancelled':
        return Colors.red.shade50;
      case 'Pending':
      default:
        return Colors.orange.shade50;
    }
  }

  Future<void> _showComplaintDialog(Map<String, dynamic> booking) async {
    final descriptionCtrl = TextEditingController();

    String category = 'Maintenance';
    String severity = 'medium';
    bool submitting = false;

    await showDialog<void>(
      context: context,
      barrierDismissible: !submitting,
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            Future<void> submitComplaint() async {
              final description = descriptionCtrl.text.trim();

              if (description.isEmpty) {
                ScaffoldMessenger.of(this.context).showSnackBar(
                  const SnackBar(
                    content: Text('Please describe your complaint first.'),
                    behavior: SnackBarBehavior.floating,
                  ),
                );
                return;
              }

              final hostelId = booking['hostelId'] as int;
              if (hostelId <= 0) {
                ScaffoldMessenger.of(this.context).showSnackBar(
                  const SnackBar(
                    content: Text('Could not determine hostel for this booking.'),
                    behavior: SnackBarBehavior.floating,
                  ),
                );
                return;
              }

              setDialogState(() => submitting = true);

              try {
                await Api().fileComplaint(
                  hostelId: hostelId,
                  category: category,
                  severity: severity,
                  description: description,
                );

                if (!mounted) return;

                Navigator.of(dialogContext).pop();

                ScaffoldMessenger.of(this.context).showSnackBar(
                  SnackBar(
                    content: Text(
                      'Complaint submitted successfully.',
                      style: GoogleFonts.dmSans(),
                    ),
                    backgroundColor: AppTheme.tealDeep,
                    behavior: SnackBarBehavior.floating,
                    margin: const EdgeInsets.all(16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                );
              } catch (e) {
                if (!mounted) return;

                setDialogState(() => submitting = false);

                ScaffoldMessenger.of(this.context).showSnackBar(
                  SnackBar(
                    content: Text(
                      'Complaint could not be submitted: $e',
                      style: GoogleFonts.dmSans(),
                    ),
                    backgroundColor: Colors.red.shade600,
                    behavior: SnackBarBehavior.floating,
                    margin: const EdgeInsets.all(16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                );
              }
            }

            return AlertDialog(
              backgroundColor: AppTheme.card,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(22),
              ),
              titlePadding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
              contentPadding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
              actionsPadding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
              title: Row(
                children: [
                  Container(
                    width: 40,
                    height: 40,
                    decoration: BoxDecoration(
                      color: AppTheme.teal.withOpacity(0.10),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.report_problem_outlined,
                      color: AppTheme.teal,
                      size: 20,
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Submit Complaint',
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 17,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                  ),
                ],
              ),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      booking['hostel'] as String,
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.teal,
                      ),
                    ),
                    const SizedBox(height: 14),
                    Text(
                      'Category',
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    DropdownButtonFormField<String>(
                      value: category,
                      decoration: InputDecoration(
                        filled: true,
                        fillColor: AppTheme.bg,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: AppTheme.border),
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: AppTheme.border),
                        ),
                      ),
                      items: const [
                        DropdownMenuItem(
                          value: 'Maintenance',
                          child: Text('Maintenance'),
                        ),
                        DropdownMenuItem(
                          value: 'Internet',
                          child: Text('Internet / WiFi'),
                        ),
                        DropdownMenuItem(
                          value: 'Cleanliness',
                          child: Text('Cleanliness'),
                        ),
                        DropdownMenuItem(
                          value: 'Security',
                          child: Text('Security'),
                        ),
                        DropdownMenuItem(
                          value: 'Food',
                          child: Text('Food / Mess'),
                        ),
                        DropdownMenuItem(
                          value: 'Other',
                          child: Text('Other'),
                        ),
                      ],
                      onChanged: submitting
                          ? null
                          : (value) {
                              if (value != null) {
                                setDialogState(() => category = value);
                              }
                            },
                    ),
                    const SizedBox(height: 14),
                    Text(
                      'Severity',
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    DropdownButtonFormField<String>(
                      value: severity,
                      decoration: InputDecoration(
                        filled: true,
                        fillColor: AppTheme.bg,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: AppTheme.border),
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: AppTheme.border),
                        ),
                      ),
                      items: const [
                        DropdownMenuItem(value: 'low', child: Text('Low')),
                        DropdownMenuItem(value: 'medium', child: Text('Medium')),
                        DropdownMenuItem(value: 'high', child: Text('High')),
                        DropdownMenuItem(value: 'urgent', child: Text('Urgent')),
                      ],
                      onChanged: submitting
                          ? null
                          : (value) {
                              if (value != null) {
                                setDialogState(() => severity = value);
                              }
                            },
                    ),
                    const SizedBox(height: 14),
                    Text(
                      'Description',
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    TextField(
                      controller: descriptionCtrl,
                      enabled: !submitting,
                      maxLines: 4,
                      maxLength: 1000,
                      decoration: InputDecoration(
                        hintText: 'Describe the issue clearly...',
                        hintStyle: GoogleFonts.dmSans(
                          color: AppTheme.textMuted,
                          fontSize: 12,
                        ),
                        filled: true,
                        fillColor: AppTheme.bg,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: AppTheme.border),
                        ),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: AppTheme.border),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed:
                      submitting ? null : () => Navigator.of(dialogContext).pop(),
                  child: Text(
                    'Cancel',
                    style: GoogleFonts.dmSans(
                      color: AppTheme.textMuted,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: submitting ? null : submitComplaint,
                  icon: submitting
                      ? const SizedBox(
                          width: 15,
                          height: 15,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.send_rounded, size: 16),
                  label: Text(
                    submitting ? 'Submitting...' : 'Submit',
                    style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal,
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(11),
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );

    descriptionCtrl.dispose();
  }

  void _showBookingDetails(Map<String, dynamic> booking) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: AppTheme.card,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 42,
                  height: 4,
                  decoration: BoxDecoration(
                    color: AppTheme.border,
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                const SizedBox(height: 18),
                Text(
                  booking['hostel'] as String,
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 20,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                _detailRow('Booking ID', '#${booking['id']}'),
                _detailRow('Location', booking['city'] as String),
                _detailRow('Room', booking['room'] as String),
                _detailRow('Check-in', booking['checkIn'] as String),
                _detailRow('Check-out', booking['checkOut'] as String),
                _detailRow('Duration', booking['duration'] as String),
                _detailRow('Status', booking['status'] as String),
                if (booking['statusRaw'] == 'pending') ...[
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.pop(context);
                        _activateHostelAccess(booking);
                      },
                      icon: const Icon(Icons.key_rounded),
                      label: const Text('Enter Hostel Password'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.teal,
                        foregroundColor: Colors.white,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _activateHostelAccess(Map<String, dynamic> booking) async {
    final passwordCtrl = TextEditingController();
    var submitting = false;
    await showDialog<void>(
      context: context,
      builder: (dialogContext) => StatefulBuilder(
        builder: (context, setDialogState) => AlertDialog(
          title: const Text('Hostel access password'),
          content: TextField(
            controller: passwordCtrl,
            obscureText: true,
            decoration: const InputDecoration(
              labelText: 'Password shared by owner',
              prefixIcon: Icon(Icons.key_rounded),
            ),
          ),
          actions: [
            TextButton(onPressed: submitting ? null : () => Navigator.pop(dialogContext), child: const Text('Cancel')),
            ElevatedButton(
              onPressed: submitting
                  ? null
                  : () async {
                      if (passwordCtrl.text.trim().isEmpty) return;
                      setDialogState(() => submitting = true);
                      try {
                        await Api().activateHostelAccess(
                          bookingId: booking['id'] as int,
                          password: passwordCtrl.text.trim(),
                        );
                        if (!mounted) return;
                        Navigator.pop(dialogContext);
                        await _showRoomAndBedSelection(booking);
                      } catch (error) {
                        if (mounted) {
                          ScaffoldMessenger.of(this.context).showSnackBar(
                            SnackBar(content: Text('Password verify nahi hua: $error')),
                          );
                        }
                      } finally {
                        if (context.mounted) setDialogState(() => submitting = false);
                      }
                    },
              child: Text(submitting ? 'Checking...' : 'Continue'),
            ),
          ],
        ),
      ),
    );
    passwordCtrl.dispose();
  }

  Future<void> _showRoomAndBedSelection(Map<String, dynamic> booking) async {
    final future = Api().getStudentHostelRooms(booking['hostelId'] as int);
    List<Map<String, dynamic>>? aiMatches;
    var matching = false;
    String? matchError;
    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppTheme.card,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (sheetContext) => SafeArea(
        child: SizedBox(
          height: MediaQuery.of(sheetContext).size.height * .72,
          child: StatefulBuilder(
            builder: (modalContext, setModalState) => FutureBuilder<List<dynamic>>(
              future: future,
              builder: (context, snapshot) {
                if (snapshot.hasError) {
                  return Center(child: Text('Rooms load nahi huay: ${snapshot.error}'));
                }
                if (!snapshot.hasData) {
                  return const Center(child: CircularProgressIndicator(color: AppTheme.teal));
                }
                final rooms = snapshot.data!
                    .whereType<Map>()
                    .map((row) => Map<String, dynamic>.from(row))
                    .toList();
                if (rooms.isEmpty) {
                  return const Center(child: Text('No available rooms right now.'));
                }
                final displayedRooms = aiMatches ?? rooms;
                return ListView(
                  padding: const EdgeInsets.all(20),
                  children: [
                    Text(
                      'Choose floor, room & bed',
                      style: GoogleFonts.playfairDisplay(fontSize: 22, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      'Tap an available bed to confirm it and enter your Hostel Hub.',
                      style: GoogleFonts.dmSans(color: AppTheme.textMuted),
                    ),
                    const SizedBox(height: 14),
                    SizedBox(
                      width: double.infinity,
                      child: OutlinedButton.icon(
                        onPressed: matching
                            ? null
                            : () async {
                                setModalState(() {
                                  matching = true;
                                  matchError = null;
                                });
                                try {
                                  final rawMatches = await Api().getStudentRoomMatches(
                                    booking['id'] as int,
                                  );
                                  setModalState(() {
                                    aiMatches = rawMatches
                                        .whereType<Map>()
                                        .map((row) => Map<String, dynamic>.from(row))
                                        .toList();
                                  });
                                } catch (error) {
                                  setModalState(() => matchError = error.toString());
                                } finally {
                                  if (modalContext.mounted) {
                                    setModalState(() => matching = false);
                                  }
                                }
                              },
                        icon: matching
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Icon(Icons.auto_awesome_rounded),
                        label: Text(matching ? 'Finding your best room...' : 'AI Powered Match'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: AppTheme.teal,
                          side: const BorderSide(color: AppTheme.teal),
                          padding: const EdgeInsets.symmetric(vertical: 13),
                        ),
                      ),
                    ),
                    if (matchError != null) ...[
                      const SizedBox(height: 8),
                      Text(matchError!, style: GoogleFonts.dmSans(fontSize: 11, color: Colors.red.shade700)),
                    ],
                    if (aiMatches != null) ...[
                      const SizedBox(height: 12),
                      Text(
                        'AI ranking uses your saved preferences, live availability, room capacity and monthly rent.',
                        style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.teal, height: 1.4),
                      ),
                    ],
                    const SizedBox(height: 16),
                    ...displayedRooms.map((room) {
                      final beds = (room['beds'] as List? ?? [])
                          .whereType<Map>()
                          .map((bed) => Map<String, dynamic>.from(bed))
                          .toList();
                      final score = room['match_score'];
                      return Card(
                        child: Padding(
                          padding: const EdgeInsets.all(14),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Expanded(
                                    child: Text(
                                      '${room['floor_label']} · Room ${room['room_number']}',
                                      style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                                    ),
                                  ),
                                  if (score != null)
                                    Container(
                                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                      decoration: BoxDecoration(
                                        color: AppTheme.teal.withOpacity(0.10),
                                        borderRadius: BorderRadius.circular(20),
                                      ),
                                      child: Text(
                                        '$score% match',
                                        style: GoogleFonts.dmSans(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w700,
                                          color: AppTheme.teal,
                                        ),
                                      ),
                                    ),
                                ],
                              ),
                              const SizedBox(height: 3),
                              Text(
                                '${room['room_type']} · PKR ${room['monthly_rent']}/month',
                                style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted),
                              ),
                              if (room['match_reasons'] is List) ...[
                                const SizedBox(height: 8),
                                Text(
                                  (room['match_reasons'] as List).take(2).join(' · '),
                                  style: GoogleFonts.dmSans(fontSize: 10.5, color: AppTheme.teal),
                                ),
                              ],
                              const SizedBox(height: 10),
                              Wrap(
                                spacing: 8,
                                runSpacing: 8,
                                children: beds
                                    .where((bed) => bed['status'] != 'occupied')
                                    .map(
                                      (bed) => ActionChip(
                                        label: Text('Confirm Bed ${bed['bed_number']}'),
                                        onPressed: () async {
                                          try {
                                            await Api().selectStudentBed(
                                              bookingId: booking['id'] as int,
                                              roomId: _toInt(room['id']),
                                              bedNumber: bed['bed_number'].toString(),
                                            );
                                            if (!mounted) return;
                                            Navigator.pop(sheetContext);
                                            Navigator.of(this.context).pushAndRemoveUntil(
                                              MaterialPageRoute(
                                                builder: (_) => const StudentHostelAccessScreen(
                                                  openHubImmediately: true,
                                                ),
                                              ),
                                              (_) => false,
                                            );
                                          } catch (error) {
                                            if (mounted) {
                                              ScaffoldMessenger.of(this.context).showSnackBar(
                                                SnackBar(content: Text('Bed select nahi hua: $error')),
                                              );
                                            }
                                          }
                                        },
                                      ),
                                    )
                                    .toList(),
                              ),
                            ],
                          ),
                        ),
                      );
                    }),
                  ],
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 12,
                color: AppTheme.textMuted,
              ),
            ),
          ),
          Flexible(
            child: Text(
              value,
              style: GoogleFonts.dmSans(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: AppTheme.textPrimary,
              ),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        appBar: AppBar(
          title: const Text('My Bookings'),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back_rounded),
            onPressed: () => Navigator.pop(context),
          ),
          bottom: TabBar(
            controller: _tabs,
            labelColor: Colors.white,
            unselectedLabelColor: Colors.white60,
            indicatorColor: AppTheme.tealMint,
            tabs: const [
              Tab(text: 'All'),
              Tab(text: 'Pending'),
              Tab(text: 'Confirmed'),
              Tab(text: 'Past'),
            ],
          ),
        ),
        body: _buildBody(),
      ),
    );
  }

  Widget _buildBody() {
    if (_loading) {
      return const Center(
        child: CircularProgressIndicator(color: AppTheme.teal),
      );
    }

    if (_error != null) {
      return RefreshIndicator(
        onRefresh: _loadBookings,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(24),
          children: [
            const SizedBox(height: 100),
            Icon(
              Icons.cloud_off_rounded,
              size: 58,
              color: Colors.red.shade300,
            ),
            const SizedBox(height: 14),
            Text(
              'Could not load bookings',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: AppTheme.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              _error!,
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 11,
                color: AppTheme.textMuted,
                height: 1.5,
              ),
            ),
            const SizedBox(height: 18),
            Center(
              child: ElevatedButton.icon(
                onPressed: _loadBookings,
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Try Again'),
              ),
            ),
          ],
        ),
      );
    }

    return TabBarView(
      controller: _tabs,
      children: [
        _buildList(_filtered('All')),
        _buildList(_filtered('Pending')),
        _buildList(_filtered('Confirmed')),
        _buildList(_filtered('Past')),
      ],
    );
  }

  Widget _buildList(List<Map<String, dynamic>> items) {
    if (items.isEmpty) {
      return RefreshIndicator(
        onRefresh: _loadBookings,
        child: ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          children: [
            const SizedBox(height: 130),
            Icon(
              Icons.hotel_outlined,
              size: 60,
              color: AppTheme.textMuted.withOpacity(0.3),
            ),
            const SizedBox(height: 14),
            Text(
              'No bookings here',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                color: AppTheme.textMuted,
                fontSize: 15,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              'Your database bookings will appear here',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                color: AppTheme.textMuted,
                fontSize: 12,
              ),
            ),
          ],
        ),
      );
    }

    return RefreshIndicator(
      onRefresh: _loadBookings,
      child: ListView.builder(
        physics: const AlwaysScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
        itemCount: items.length,
        itemBuilder: (_, i) => _bookingCard(items[i]),
      ),
    );
  }

  Widget _bookingCard(Map<String, dynamic> b) {
    final status = b['status'] as String;
    final color = _statusColor(status);
    final bgColor = _statusBgColor(status);
    final isConfirmed = b['statusRaw'] == 'confirmed';

    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.border, width: 0.5),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 12),
            child: Row(
              children: [
                Container(
                  width: 46,
                  height: 46,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [AppTheme.tealDeep, AppTheme.teal],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(
                    Icons.apartment_rounded,
                    color: Colors.white,
                    size: 22,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        b['hostel'] as String,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          fontSize: 15,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Row(
                        children: [
                          const Icon(
                            Icons.location_on_rounded,
                            size: 12,
                            color: AppTheme.textMuted,
                          ),
                          const SizedBox(width: 3),
                          Expanded(
                            child: Text(
                              b['city'] as String,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.dmSans(
                                fontSize: 11,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: bgColor,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    status,
                    style: GoogleFonts.dmSans(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: color,
                    ),
                  ),
                ),
              ],
            ),
          ),
          Divider(height: 0.5, color: AppTheme.border, thickness: 0.5),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: Row(
              children: [
                Expanded(
                  child: _detailCol(
                    Icons.meeting_room_rounded,
                    'Room',
                    b['room'] as String,
                  ),
                ),
                Expanded(
                  child: _detailCol(
                    Icons.event_rounded,
                    'Check-in',
                    b['checkIn'] as String,
                  ),
                ),
                Expanded(
                  child: _detailCol(
                    Icons.event_available_rounded,
                    'Check-out',
                    b['checkOut'] as String,
                  ),
                ),
              ],
            ),
          ),
          Container(
            margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: AppTheme.bg,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.receipt_long_rounded,
                  size: 14,
                  color: AppTheme.textMuted,
                ),
                const SizedBox(width: 6),
                Text(
                  'Booking #${b['id']}',
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    color: AppTheme.textMuted,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const Spacer(),
                if ((b['total'] as int) > 0)
                  Text(
                    'PKR ${_money(b['total'] as int)}',
                    style: GoogleFonts.dmSans(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.teal,
                    ),
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () => _showBookingDetails(b),
                    icon: const Icon(Icons.info_outline_rounded, size: 15),
                    label: Text(
                      'Details',
                      style: GoogleFonts.dmSans(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppTheme.teal,
                      side: const BorderSide(
                        color: AppTheme.teal,
                        width: 0.5,
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 10),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                  ),
                ),
                if (isConfirmed) ...[
                  const SizedBox(width: 10),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: () => _showComplaintDialog(b),
                      icon: const Icon(
                        Icons.report_problem_outlined,
                        size: 15,
                      ),
                      label: Text(
                        'Complaint',
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: AppTheme.tealDeep,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        padding: const EdgeInsets.symmetric(vertical: 10),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _money(int value) {
    return value.toString().replaceAllMapped(
          RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
          (m) => '${m[1]},',
        );
  }

  Widget _detailCol(IconData icon, String label, String value) {
    return Column(
      children: [
        Icon(icon, size: 16, color: AppTheme.teal),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: AppTheme.textPrimary,
          ),
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.center,
        ),
        Text(
          label,
          style: GoogleFonts.dmSans(
            fontSize: 10,
            color: AppTheme.textMuted,
          ),
        ),
      ],
    );
  }
}
